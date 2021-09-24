import argparse
from models.core import blocks
import time
import datetime
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
import torch.distributed as dist
import models
from utils import *
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import RandomAdjustSharpness, RandomGrayscale, RandomHorizontalFlip, RandomResizedCrop, RandomRotation

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

mean = [0.5071, 0.4865, 0.4409]
std = [0.2673, 0.2564, 0.2762]


def train(train_loader, model, criterion, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.cuda(
            non_blocking=True), labels.cuda(non_blocking=True)

        origin_labels = labels
        
        if args.mixup or args.label_smoothing:
            labels = one_hot(labels, 100)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(images)
            loss = criterion(output, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        acc1, acc5 = accuracy(output, origin_labels, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.local_rank == 0:
            logger.info(f'#{epoch:>3} [{args.local_rank}:{i:>3}/{len(train_loader)}], '
                        f't={batch_time.val:>.3f}/{batch_time.avg:>.3f}, '
                        f't1={top1.val:>6.3f}/{top1.avg:>6.3f}, '
                        f't5={top5.avg:>6.3f}, '
                        f'lr={optimizer.param_groups[0]["lr"]:>.8f}, '
                        f'l={losses.avg:>.3f}')


def validate(test_loader, model, criterion, args):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    losses = 0

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.cuda(
                non_blocking=True), labels.cuda(non_blocking=True)
            output = model(images)
            loss = criterion(output, labels)
            losses += loss

            acc1, acc5 = accuracy(output.data, labels, topk=(1, 5))

            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     logger.info(f'Validation [{args.local_rank}:{i:>2}/{len(test_loader)}], '
            #                 f'time={batch_time.val:>.3f}({batch_time.avg:>.3f}), '
            #                 f'loss={loss.item():>.5f}, '
            #                 f'top1={top1.val:>6.3f}({top1.avg:>6.3f}), '
            #                 f'top5={top5.val:>6.3f}({top5.avg:>6.3f})')

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
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    model_names += sorted(name for name in torchvision.models.__dict__
                          if name.islower() and not name.startswith("__")
                          and callable(torchvision.models.__dict__[name]))

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/datasets/ILSVRC2012',
                        help='path to the ImageNet dataset.')
    parser.add_argument('--torch', action='store_true',
                        help='use torchvision models. (default: false)')
    parser.add_argument('--model', type=str, default='muxnet_v2', choices=model_names,
                        help='type of model to use. (default: muxnet_v2)')
    parser.add_argument('--input-size', type=int, default=224,  metavar='SIZE',
                        help='size of the input image size. (default: 224)')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pre-trained model. (default: false)')
    parser.add_argument('--deterministic', action='store_true',
                        help='reproducibility. (default: false)')
    parser.add_argument('--local_rank', type=int, default=0, metavar='RANK')
    parser.add_argument('--workers', '-j', type=int, default=4, metavar='N',
                        help='number of data loading workers pre GPU. (default: 4)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='mini-batch size, this is the total batch size of all GPUs. (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,  metavar='N',
                        help='number of total epochs to run. (default: 100)')
    parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'rmsprop'],
                        help='optimizer. (default: sgd)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate. (default: 0.1)')
    parser.add_argument('--rmsprop-decay', type=float, default=0.9, metavar='D',
                        help='decay of RMSprop. (default: 0.9)')
    parser.add_argument('--rmsprop-epsilon', type=float,
                        default=1e-8, metavar='E')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum of SGD. (default: 0.9)')
    parser.add_argument('--nesterov', action='store_true',
                        help='nesterov of SGD. (default: false)')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='weight decay. (default: 1e-4)')
    parser.add_argument('--lr-mode', type=str, default='cosine', choices=['step', 'cosine'],
                        help="learning rate scheduler mode, options are [cosine, step]. (default: cosine)")
    parser.add_argument('--lr-decay', type=float, default=0.1, metavar='RATE',
                        help='decay rate of learning rate. (default: 0.1)')
    parser.add_argument('--lr-decay-epochs', type=int, default=0, metavar='N',
                        help='interval for periodic learning rate decays. (default: 0)')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='number of warmup epochs. (default: 0)')
    parser.add_argument('--mixup', action='store_true',
                        help='whether train the model with mix-up. (default: false)')
    parser.add_argument('--augment', action='store_true',
                        help='data augmentation. (default: false)')
    parser.add_argument('--mixup-alpha', type=float, default=0.2, metavar='V',
                        help='beta distribution parameter for mixup sampling. (default: 0.2)')
    parser.add_argument('--mixup-off-epoch', type=int, default=0, metavar='N',
                        help='how many last epochs to train without mixup. (default: 0)')
    parser.add_argument('--label-smoothing', action='store_true',
                        help='use label smoothing or not in training. (default: false)')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
    parser.add_argument('--bn-momentum', type=float, default=0.1, metavar='M')
    parser.add_argument('--bn-epsilon', type=float, default=1e-5, metavar='E')
    parser.add_argument('--bn-position', type=str, default='before', choices=['before', 'after', 'none'],
                        help='norm layer / activation layer. (default: before)')
    parser.add_argument('--moving-average-decay',
                        type=float, default=0.9999, metavar='M')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate model on validation set. (default: false)')
    parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                        help='print frequency. (default: 10)')
    parser.add_argument('--sync_bn', action='store_true',
                        help='use SyncBatchNorm. (default: false)')
    parser.add_argument('--amp', action='store_true',
                        help='mixed precision. (default: false)')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--dali-cpu', action='store_true',
                        help='runs CPU based version of DALI pipeline. (default: false)')
    parser.add_argument('--output-dir', type=str,
                        default=f'logs/{datetime.date.today()}', metavar='DIR')
    return parser.parse_args()


if __name__ == '__main__':
    assert torch.cuda.is_available(), 'CUDA IS NOT AVAILABLE!!'

    args = parse_args()
    args.batch_size = int(args.batch_size / torch.cuda.device_count())

    torch.backends.cudnn.benchmark = True
    if args.deterministic:
        manual_seed(args.local_rank)

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group('nccl')

    logger = make_logger(
        f'cifar100_{args.model}', f'{args.output_dir}/{args.model}', rank=args.local_rank)
    logger.info(args)

    if args.local_rank == 0 and args.download:
        CIFAR100(args.data_dir, True, download=True)
        CIFAR100(args.data_dir, False, download=True)
    dist.barrier()

    train_dataset = CIFAR100(
        args.data_dir,
        True,
        transform=T.Compose([
            # T.RandomAffine((-15, 15), translate=(0.1, 0.1), scale=(0.8, 1.2)),
            T.RandomCrop(32, 4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
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

    test_dataset = CIFAR100(
        args.data_dir,
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

    with blocks.batchnorm(args.bn_momentum, args.bn_epsilon, args.bn_position):
        if args.torch:
            model = torchvision.models.__dict__[args.model]()
        else:
            model = models.__dict__[args.model](thumbnail=True, num_classes=100)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.cuda()
    if args.local_rank == 0:
        logger.info(f'Model: \n{model}')

    model = DistributedDataParallel(model, device_ids=[args.local_rank])

    if args.no_wd:
        wd = []
        no_wd = []
        for m, n, p in module_parameters(model):
            if isinstance(m, nn.modules.batchnorm._BatchNorm) or n == 'bias':
                no_wd.append(p)
            else:
                wd.append(p)

        assert len(list(model.parameters())) == (len(no_wd) + len(wd)), ''

        if args.optim == 'sgd':
            optimizer = torch.optim.SGD([
                {'params': wd, 'weight_decay': args.wd},
                {'params': no_wd, 'weight_decay': 0.}],
                args.lr,
                momentum=args.momentum,
                nesterov=args.nesterov)
        elif args.optim == 'rmsprop':
            optimizer = torch.optim.RMSprop([
                {'params': wd, 'weight_decay': args.wd},
                {'params': no_wd, 'weight_decay': 0.}],
                lr=args.lr,
                alpha=args.rmsprop_decay,
                momentum=args.momentum,
                eps=args.rmsprop_epsilon)
        else:
            raise ValueError(f'Invalid optimizer name: {args.optim}.')
    else:
        if args.optim == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),
                                        args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.wd,
                                        nesterov=args.nesterov)
        elif args.optim == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(),
                                            lr=args.lr,
                                            alpha=args.rmsprop_decay,
                                            momentum=args.momentum,
                                            weight_decay=args.wd,
                                            eps=args.rmsprop_epsilon)
        else:
            raise ValueError(f'Invalid optimizer name: {args.optim}.')

    if args.label_smoothing or args.mixup:
        criterion = LabelSmoothingCrossEntropyLoss(0.1).cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()

    val_criterion = torch.nn.CrossEntropyLoss().cuda()

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    benchmark = Benchmark()
    if args.evaluate:
        validate(test_loader, model, val_criterion)
    else:
        if args.lr_mode == 'step':
            scheduler = lr.WarmUpStepLR(
                optimizer,
                warmup_steps=args.warmup_epochs * len(train_loader),
                step_size=args.lr_decay_epochs * len(train_loader),
                gamma=args.lr_decay
            )
        else:
            scheduler = lr.WarmUpCosineLR(
                optimizer,
                warmup_steps=args.warmup_epochs * len(train_loader),
                steps=args.epochs * len(train_loader),
                min_lr=0
            )

        if args.local_rank == 0:
            logger.info(scheduler)

        for epoch in range(0, args.epochs):
            train_sampler.set_epoch(epoch)
            train(train_loader, model, criterion,
                  optimizer, scheduler, epoch, args)
            validate(test_loader, model, val_criterion, args)

        if args.local_rank == 0:
            model_name = f'{args.output_dir}/{args.model}/{args.model}_{time.time()}.pth'
            # with moving_average.average_parameters():
            torch.save(model.module.state_dict(), model_name)
            logger.info(f'Saved: {model_name}!')
    if args.local_rank == 0:
        logger.info(f'Total time: {benchmark.elapsed():>.3f}s')
