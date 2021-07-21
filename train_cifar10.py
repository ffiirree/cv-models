import argparse
import time
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T
import torch.distributed as dist
from torchvision.transforms.transforms import RandomAdjustSharpness, RandomGrayscale, RandomHorizontalFlip, RandomResizedCrop, RandomRotation
import models
from utils import *
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    model.train()
    
    end = time.time()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(images)
            loss = criterion(output, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(f'#{epoch:>3} [{args.local_rank}:{i:>3}/{len(train_loader)}], '
                        f'lr={optimizer.param_groups[0]["lr"]:>.10f}, '
                        f't={batch_time.val:>.3f}/{batch_time.avg:>.3f}, '
                        f'l={losses.val:>.5f}/{losses.avg:>.5f}, '
                        f't1={top1.val:>6.3f}/{top1.avg:>6.3f}, '
                        f't5={top5.val:>6.3f}/{top5.avg:>6.3f}')


def test(test_loader, model, epoch, args):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    losses = 0

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            output = model(images)
            loss = criterion(output, labels)
            losses += loss
            
            acc1, acc5 = accuracy(output.data, labels, topk=(1, 5))

            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info(f'Validation [{args.local_rank}:{i:>2}/{len(test_loader)}], '
                        f'time={batch_time.val:>.3f}({batch_time.avg:>.3f}), '
                        f'loss={loss.item():>.5f}, '
                        f'top1={top1.val:>6.3f}({top1.avg:>6.3f}), '
                        f'top5={top5.val:>6.3f}({top5.avg:>6.3f})')

    dist.all_reduce(losses)
    top1 = torch.tensor([top1.avg]).cuda()
    top5 = torch.tensor([top5.avg]).cuda()
    dist.all_reduce(top1)
    dist.all_reduce(top5)
    if args.local_rank == 0:
        logger.info(f'loss={losses.item() / (len(test_loader) * dist.get_world_size()):>.5f}, '
                    f'top1={top1.item() / dist.get_world_size():>6.3f}, '
                    f'top5={top5.item() / dist.get_world_size():>6.3f}')

def parse_args():
    model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))
        
    parser = argparse.ArgumentParser()
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='XNet',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: XNet)')
    parser.add_argument('--amp',                action='store_true')
    parser.add_argument('--local_rank',         type=int, default=0)
    parser.add_argument('-j', '--workers',      type=int,   default=8)
    parser.add_argument('--epochs',             type=int,   default=90)
    parser.add_argument('-b', '--batch-size',   type=int,   default=512)
    parser.add_argument('--filters',            metavar='FILTERS', type=int, default=32)
    parser.add_argument('--lr',                 type=float, default=0.1)
    parser.add_argument('--momentum',           type=float, default=0.9)
    parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 0.0005)')
    parser.add_argument('--download',           action='store_true')
    parser.add_argument('--output-dir',         type=str,   default='logs')
    parser.add_argument('--print-freq', '-p', default=20, type=int,
                        metavar='N', help='print frequency (default: 10)')
    return parser.parse_args()

if __name__ == '__main__':
    assert torch.cuda.is_available(), 'CUDA IS NOT AVAILABLE!!'
  
    args = parse_args()
    logger = make_logger(f'cifar10_{args.arch}', 'logs')

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group('nccl')
    device = torch.device(f'cuda:{args.local_rank}')

    logger.info(args)
    
    if args.local_rank == 0 and args.download:
        CIFAR10(args.data, True, download=True)
        CIFAR10(args.data, False, download=True)
    dist.barrier()

    train_dataset = CIFAR10(
        args.data, 
        True,
        transform=T.Compose([
            # T.RandomAffine((-15, 15), translate=(0.1, 0.1), scale=(0.8, 1.2)),
            # T.RandomGrayscale(),
            # T.RandomAdjustSharpness(2),
            T.RandomCrop(32, 4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            
            # T.RandomErasing(p=0.25, scale=(0.04, 0.15))
        ])
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler
    )
    
    test_dataset = CIFAR10(
        args.data, 
        False, 
        transform=T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    )
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=test_sampler
    )

    net = models.__dict__[args.arch](3, 10)

    if args.local_rank == 0:
        logger.info(f'Model: \n{net}')
        params_num = sum(p.numel() for p in net.parameters())
        logger.info(f'Params: {params_num} ({params_num / 1000000:>7.4f}M)')
    net.to(device)
    # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    benchmark = Benchmark()
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train(train_loader, net, criterion, optimizer, epoch, args)
        test(test_loader, net, epoch, args)
        scheduler.step()
    logger.info(f'{benchmark.elapsed():>.3f}')

    # if args.local_rank == 0:
    #     model_name = f'{args.output_dir}/cifar10_{args.arch}_{time.time()}.pt'
    #     torch.save(net.module.state_dict(), model_name)
    #     logger.info(f'Saved: {model_name}!')
