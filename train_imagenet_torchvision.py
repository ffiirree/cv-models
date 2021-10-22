import os
import time
import datetime
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from utils import *
import models
from models.core import blocks


def parse_args():
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    model_names += sorted(name for name in torchvision.models.__dict__
                          if name.islower() and not name.startswith("__")
                          and callable(torchvision.models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
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
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--deterministic', action='store_true',
                        help='reproducibility. (default: false)')
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
    parser.add_argument('--augment', type=str, default='none',
                        choices=['none', 'standard', 'randaugment', 'autoaugment'])
    parser.add_argument('--randaugment_n', type=int, default=2)
    parser.add_argument('--randaugment_m', type=int, default=10)
    parser.add_argument('--dropout-rate', type=float, default=0.)
    parser.add_argument('--drop-path-rate', type=float, default=0.)
    parser.add_argument('--output-dir', type=str,
                        default=f'logs/{datetime.date.today()}', metavar='DIR')
    return parser.parse_args()


def train(train_loader, model, criterion, optimizer, scheduler, moving_average, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    mixup = MixUp(args.mixup_alpha)

    use_mixup = args.mixup and args.mixup_off_epoch < (args.epochs - epoch)
    if args.local_rank == 0 and use_mixup:
        logger.info(f'mixup: {use_mixup}, {mixup}')

    model.train()

    end = time.time()
    for i, (images, labels) in enumerate(train_loader):
        input = images.cuda(non_blocking=True)
        target = labels.cuda(non_blocking=True)
        original_target = target

        # MixUP
        if args.mixup or args.label_smoothing:
            target = one_hot(target, 1000)

        if use_mixup:
            input, target = mixup.mix(input, target)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(input)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # moving_average.update()

        scheduler.step()

        acc1, acc5 = accuracy(output, original_target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and i != 0:
            logger.info(f'#{epoch:>3} [{args.local_rank}:{i:>4}], '
                        f't={batch_time.val:>.3f}/{batch_time.avg:>.3f}, '
                        f't1={top1.val:>6.3f}/{top1.avg:>6.3f}, '
                        f't5={top5.val:>6.3f}/{top5.avg:>6.3f}, '
                        f'lr={optimizer.param_groups[0]["lr"]:>.6f}, '
                        f'l={losses.avg:>.3f}')


def validate(val_loader, model, criterion, moving_average):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    losses = 0

    model.eval()
    end = time.time()
    for i, (images, labels) in enumerate(val_loader):
        input = images.cuda(non_blocking=True)
        target = labels.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():  # , moving_average.average_parameters():
            output = model(input)
            loss = criterion(output, target)
            losses += loss

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

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
    args.workers = args.workers // torch.cuda.device_count()

    torch.backends.cudnn.benchmark = True
    if args.deterministic:
        manual_seed(args.local_rank)

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group('nccl')

    logger = make_logger(
        f'imagenet_{args.model}', f'{args.output_dir}/{args.model}', rank=args.local_rank)
    logger.info(args)

    with blocks.batchnorm(args.bn_momentum, args.bn_epsilon, args.bn_position):
        if args.torch:
            model = torchvision.models.__dict__[args.model](pretrained=args.pretrained)
        else:
            model = models.__dict__[args.model](
                pretrained=args.pretrained, 
                pth=args.path,
                dropout_rate=args.dropout_rate,
                drop_path_rate=args.drop_path_rate
            )
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.cuda()
    if args.local_rank == 0 and not args.evaluate:
        logger.info(f'Model: \n{model}')

    model = DistributedDataParallel(model, device_ids=[args.local_rank])

    param_groups = group_params(model, wd=args.wd, no_bias_decay=args.no_wd)
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(param_groups,
                                    args.lr,
                                    momentum=args.momentum,
                                    nesterov=args.nesterov)
    elif args.optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(param_groups,
                                        lr=args.lr,
                                        alpha=args.rmsprop_decay,
                                        momentum=args.momentum,
                                        eps=args.rmsprop_epsilon)
    else:
        raise ValueError(f'Invalid optimizer name: {args.optim}.')

    if args.label_smoothing or args.mixup:
        criterion = LabelSmoothingCrossEntropyLoss(0.1).cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()

    val_criterion = torch.nn.CrossEntropyLoss().cuda()

    moving_average = ExponentialMovingAverage(
        model.parameters(), decay=args.moving_average_decay)

    aug = [
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip()
    ]
    if args.augment == 'randaugment':
        aug.append(RandAugment(args.randaugment_n, args.randaugment_m))
    elif args.augment == 'autoaugment':
        aug.append(T.AutoAugment(T.AutoAugmentPolicy.IMAGENET))
    elif args.augment == 'standard':
        aug.append(T.ColorJitter(brightness=0.4,
                                 contrast=0.4,
                                 saturation=0.4,
                                 hue=0.4))
    elif args.augment == 'none':
        ...
    else:
        raise ValueError(f'Invalid augmentation name: {args.augment}.')

    train_dataset = ImageFolder(
        os.path.join(args.data_dir, 'train'),
        T.Compose(aug+[
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ]))

    if args.local_rank == 0:
        logger.info(f'Transforms: \n{train_dataset.transform}')

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler
    )

    val_dataset = ImageFolder(os.path.join(args.data_dir, 'val'),
                              T.Compose([
                                  # If size is an int, smaller edge of the image will be 
                                  # matched to this number.
                                  T.Resize(256),
                                  T.CenterCrop(224),
                                  LocalDestroyer(),
                                  T.ToTensor(),
                                  T.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225]),
                              ]))

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=DistributedSampler(val_dataset, shuffle=False))

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    benchmark = Benchmark()
    if args.evaluate:
        validate(val_loader, model, val_criterion, moving_average)
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
            logger.info(f'step:{len(train_loader)}')

        for epoch in range(0, args.epochs):
            train_sampler.set_epoch(epoch)
            train(train_loader, model, criterion,
                  optimizer, scheduler, moving_average, epoch, args)
            validate(val_loader, model, val_criterion, moving_average)

            if args.local_rank == 0 and epoch > (args.epochs - 10):
                model_name = f'{args.output_dir}/{args.model}/{args.model}_{epoch:0>3}_{time.time()}.pth'
                # with moving_average.average_parameters():
                torch.save(model.module.state_dict(), model_name)
                logger.info(f'Saved: {model_name}!')
    logger.info(f'Total time: {benchmark.elapsed():>.3f}s')