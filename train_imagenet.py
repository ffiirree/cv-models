import os
import json
import time
import datetime
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist

from cvm.utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # dataset
    parser.add_argument('--data-dir', type=str, default='/datasets/ILSVRC2012',
                        help='path to the ImageNet dataset.')
    parser.add_argument('--dataset', type=str, default='ImageNet', metavar='NAME',
                        choices=list_datasets() + ['ImageNet'], help='dataset type.')
    parser.add_argument('--workers', '-j', type=int, default=4, metavar='N',
                        help='number of data loading workers pre GPU. (default: 4)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='mini-batch size, this is the total batch size of all GPUs. (default: 256)')
    parser.add_argument('--crop-size', type=int, default=224)
    parser.add_argument('--crop-padding', type=int, default=4, metavar='S')
    parser.add_argument('--val-resize-size', type=int, default=256)
    parser.add_argument('--val-crop-size', type=int, default=224)

    # model
    parser.add_argument('--model', type=str, default='muxnet_v2', choices=list_models() + list_models(True),
                        help='type of model to use. (default: muxnet_v2)')
    parser.add_argument('--torch', action='store_true',
                        help='use torchvision models. (default: false)')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pre-trained model. (default: false)')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                        help='number of label classes')
    parser.add_argument('--bn-eps', type=float, default=None)
    parser.add_argument('--bn-momentum', type=float, default=None)

    # optimizer
    parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'rmsprop'],
                        help='optimizer. (default: sgd)')
    parser.add_argument('--weight-decay', '--wd', type=float, default=1e-4,
                        help='weight decay. (default: 1e-4)')
    parser.add_argument('--no-bias-bn-wd', action='store_true',
                        help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
    parser.add_argument('--rmsprop-decay', type=float, default=0.9, metavar='D',
                        help='decay of RMSprop. (default: 0.9)')
    parser.add_argument('--rmsprop-epsilon', type=float,
                        default=1e-8, metavar='E')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum of SGD. (default: 0.9)')
    parser.add_argument('--nesterov', action='store_true',
                        help='nesterov of SGD. (default: false)')
    parser.add_argument('--adam-betas', type=list,
                        nargs='+', default=[0.9, 0.999])

    # learning rate
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate. (default: 0.1)')
    parser.add_argument('--lr-sched', type=str, default='cosine', choices=['step', 'cosine'],
                        help="learning rate scheduler mode, options are [cosine, step]. (default: cosine)")
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, metavar='RATE',
                        help='decay rate of learning rate. (default: 0.1)')
    parser.add_argument('--lr-decay-epochs', type=int, default=0, metavar='N',
                        help='interval for periodic learning rate decays. (default: 0)')
    parser.add_argument('--epochs', type=int, default=100,  metavar='N',
                        help='number of total epochs to run. (default: 100)')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='number of warmup epochs. (default: 0)')

    # augmentation | regularization
    parser.add_argument('--hflip', type=float, default=0.5, metavar='P')
    parser.add_argument('--vflip', type=float, default=0.0, metavar='P')
    parser.add_argument('--color-jitter', type=float, default=0., metavar='M')
    parser.add_argument('--random-erasing', type=float,
                        default=0., metavar='P')
    parser.add_argument('--mixup-alpha', type=float, default=0., metavar='V',
                        help='beta distribution parameter for mixup sampling. (default: 0.0)')
    parser.add_argument('--cutmix-alpha', type=float, default=0., metavar='V',
                        help='beta distribution parameter for cutmix sampling. (default: 0.0)')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='use label smoothing or not in training. (default: 0.0)')
    parser.add_argument('--augment', type=str, default=None,
                        choices=['randaugment', 'autoaugment'])
    parser.add_argument('--randaugment-n', type=int, default=2, metavar='N',
                        help='RandAugment n.')
    parser.add_argument('--randaugment-m', type=int, default=10, metavar='M',
                        help='RandAugment m.')
    parser.add_argument('--dropout-rate', type=float, default=0., metavar='P',
                        help='dropout rate. (default: 0.0)')
    parser.add_argument('--drop-path-rate', type=float, default=0., metavar='P',
                        help='drop path rate. (default: 0.0)')

    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--deterministic', action='store_true',
                        help='reproducibility. (default: false)')
    parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                        help='print frequency. (default: 10)')
    parser.add_argument('--sync_bn', action='store_true',
                        help='use SyncBatchNorm. (default: false)')
    parser.add_argument('--amp', action='store_true',
                        help='mixed precision. (default: false)')
    parser.add_argument('--dali', action='store_true',
                        help='use nvidia dali.')
    parser.add_argument('--dali-cpu', action='store_true',
                        help='runs CPU based version of DALI pipeline. (default: false)')
    parser.add_argument('--output-dir', type=str,
                        default=f'logs/{datetime.date.today()}', metavar='DIR')
    return parser.parse_args()


def train(train_loader, model, criterion, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        if args.dali:
            input = data[0]["data"]
            target = data[0]["label"].squeeze(-1).long()
        else:
            input = data[0].cuda(non_blocking=True)
            target = data[1].cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(input)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and i != 0:
            logger.info(f'#{epoch:>3} [{args.local_rank}:{i:>4}], '
                        f't={batch_time.val:>.3f}/{batch_time.avg:>.3f}, '
                        f't1={top1.val:>6.3f}/{top1.avg:>6.3f}, '
                        f't5={top5.val:>6.3f}/{top5.avg:>6.3f}, '
                        f'lr={optimizer.param_groups[0]["lr"]:>.6f}, '
                        f'l={losses.avg:>.3f}')


def validate(val_loader, model, criterion):
    top1 = AverageMeter()
    top5 = AverageMeter()

    losses = 0

    model.eval()
    for i, data in enumerate(val_loader):
        if args.dali:
            input = data[0]["data"]
            target = data[0]["label"].squeeze(-1).long()
        else:
            input = data[0].cuda(non_blocking=True)
            target = data[1].cuda(non_blocking=True)

        with torch.inference_mode():
            output = model(input)
            loss = criterion(output, target)
            losses += loss

        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

    dist.all_reduce(losses)
    top1 = torch.tensor([top1.avg]).cuda()
    top5 = torch.tensor([top5.avg]).cuda()
    dist.all_reduce(top1)
    dist.all_reduce(top5)
    if args.local_rank == 0:
        logger.info(f'loss={losses.item() / (len(val_loader) * dist.get_world_size()):>.5f}, '
                    f'top1={top1.item() / dist.get_world_size():>6.3f}, '
                    f'top5={top5.item() / dist.get_world_size():>6.3f}')


if __name__ == '__main__':
    assert torch.cuda.is_available(), 'CUDA IS NOT AVAILABLE!!'

    args = parse_args()

    args.batch_size = int(args.batch_size / torch.cuda.device_count())
    args.local_rank = int(os.environ['LOCAL_RANK'])

    torch.backends.cudnn.benchmark = True
    if args.deterministic:
        manual_seed(args.seed + args.local_rank)
        torch.use_deterministic_algorithms(True)

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group('nccl')

    logger = make_logger(
        f'imagenet_{args.model}', f'{args.output_dir}/{args.model}',
        rank=args.local_rank
    )
    if args.local_rank == 0:
        logger.info(f'Args: \n{json.dumps(vars(args), indent=4)}')

    model = create_model(
        args.model,
        torch=args.torch,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        drop_path_rate=args.drop_path_rate,
        bn_eps=args.bn_eps,
        bn_momentum=args.bn_momentum,
        thumbnail=(args.crop_size < 128),
        pretrained=args.pretrained,
        pth=args.model_path,
        sync_bn=args.sync_bn,
        distributed=True,
        local_rank=args.local_rank
    )

    optimizer = create_optimizer(args.optim, model, **dict(vars(args)))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    train_loader = create_loader(
        root=args.data_dir,
        is_training=True,
        **(dict(vars(args)))
    )
    val_loader = create_loader(
        root=args.data_dir,
        is_training=False,
        **(dict(vars(args)))
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    scheduler = create_scheduler(
        args.lr_sched,
        optimizer,
        len(train_loader),
        **(dict(vars(args)))
    )

    if args.local_rank == 0:
        logger.info(f'Model: \n{model}')
        if not args.dali:
            logger.info(f'Training: \n{train_loader.dataset.transform}')
            logger.info(f'Validation: \n{val_loader.dataset.transform}')
        logger.info(f'Optimizer: \n{optimizer}')
        logger.info(f'Criterion: {criterion}')
        logger.info(f'Scheduler: {scheduler}')
        logger.info(f'Steps/Epoch: {len(train_loader)}')

    benchmark = Benchmark()
    for epoch in range(0, args.epochs):
        if not args.dali:
            train_loader.sampler.set_epoch(epoch)
        
        train(train_loader, model, criterion, optimizer,
              scheduler, epoch, args)
        validate(val_loader, model, criterion)
        
        if args.dali:
            train_loader.reset()
            val_loader.reset()

        if args.local_rank == 0 and epoch > (args.epochs - 10):
            model_name = f'{args.output_dir}/{args.model}/{args.model}_{epoch:0>3}_{time.time()}.pth'
            torch.save(model.module.state_dict(), model_name)
            logger.info(f'Saved: {model_name}!')
    logger.info(f'Total time: {benchmark.elapsed():>.3f}s')
