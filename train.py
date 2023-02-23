import json
import time
import traceback
import datetime
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T

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
    parser.add_argument('--random-scale', type=float, nargs='+', default=[0.08, 1.0],
                        help="scale range for 'RandomReiszeCrop()'. (training stage)")
    parser.add_argument('--random-ratio', type=float, nargs='+', default=[3./4., 4./3.], 
                        help="ratio range for 'RandomResizedCrop()'. (training stage)")
    parser.add_argument('--crop-size', type=int, default=224,
                        help="crop size for 'RandomResizedCrop()'/'RandomCrop()'. (training stage)")
    parser.add_argument('--crop-padding', type=int, default=4, metavar='S',
                        help="crop padding for 'RandomCrop()'. (training stage)")
    parser.add_argument('--val-resize-size', type=int, default=256,
                        help="size for 'Resize()'. (validation stage)")
    parser.add_argument('--val-crop-size', type=int, default=224,
                        help="crop size for 'CenterCrop()'. (validation stage)")
    parser.add_argument("--interpolation", default="bilinear", type=str,
                        help="the interpolation method (default: bilinear)")

    # model
    parser.add_argument('--model', type=str, default='resnet18_v1', choices=list_models(),
                        help='type of model to use. (default: resnet18_v1)')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pre-trained model. (default: false)')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                        help='number of label classes')
    parser.add_argument('--in-channels', type=int, default=3, metavar='N',
                        help='number of image channels.')
    parser.add_argument('--bn-eps', type=float, default=None)
    parser.add_argument('--bn-momentum', type=float, default=None)

    # optimizer
    parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'rmsprop', 'adam', 'adamw'],
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
    parser.add_argument('--adam-betas', type=float,
                        nargs='+', default=[0.9, 0.999])
    parser.add_argument("--clip-grad-norm", type=float, default=None, metavar='NORM',
                        help="the maximum gradient norm (default None)")

    # learning rate
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate. (default: 0.1)')
    parser.add_argument('--lr-sched', type=str, default=None, choices=['step', 'cosine'],
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
    parser.add_argument('--random-erasing', type=float, default=0., metavar='P')
    parser.add_argument('--mixup-alpha', type=float, default=0., metavar='V',
                        help='beta distribution parameter for mixup sampling. (default: 0.0)')
    parser.add_argument('--random-frequencies-erasing', type=float, default=0., metavar='P')
    parser.add_argument('--random-gaussian-blur', type=float, nargs='+', default=None)
    parser.add_argument('--cutmix-alpha', type=float, default=0., metavar='V',
                        help='beta distribution parameter for cutmix sampling. (default: 0.0)')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='use label smoothing or not in training. (default: 0.0)')
    parser.add_argument("--ra-repetitions", default=0, type=int,
                        help="number of repetitions for Repeated Augmentation (default: 0)")
    parser.add_argument('--augment', type=str, default=None)
    parser.add_argument('--dropout-rate', type=float, default=0., metavar='P',
                        help='dropout rate. (default: 0.0)')
    parser.add_argument('--drop-path-rate', type=float, default=0., metavar='P',
                        help='drop path rate. (default: 0.0)')

    # model exponential moving average
    parser.add_argument('--model-ema', action='store_true', default=False,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model-ema-decay', type=float, default=0.99998,
                        help='decay factor for model weights moving average (default: 0.99998)')
    parser.add_argument('--model-ema-steps', type=int, default=32,
                        help='the number of iterations that controls how often to update the EMA model (default: 32)')

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


def train(train_loader, model, criterion, optimizer, scheduler, scaler, epoch, args, mixupcutmix_fn=None, model_ema: ExponentialMovingAverage = None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        if mixupcutmix_fn is not None:
            input, target = mixupcutmix_fn(input, target)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(input)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        if args.clip_grad_norm is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        if model_ema is not None and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.warmup_epochs:
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)

        end = time.time()

        if i % args.print_freq == 0 and i != 0:
            logger.info(f'#{epoch:>3}[{i:>4}] t={batch_time.avg:>.3f}, '
                        f't1={top1.avg:>6.3f}, t5={top5.avg:>6.3f}, '
                        f'lr={optimizer.param_groups[0]["lr"]:>.6f}, '
                        f'l={losses.avg:>.3f}')


def validate(val_loader, model, criterion, log_suffix=''):
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    model.eval()
    for input, target in val_loader:
        with torch.inference_mode():
            output = model(input)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))
        losses.update(loss.item(), input.size(0))

    logger.info(f'{log_suffix}loss={losses.avg:>.5f}, top1={top1.avg:>6.3f}, top5={top5.avg:>6.3f}')


if __name__ == '__main__':
    assert torch.cuda.is_available(), 'CUDA IS NOT AVAILABLE!!'

    args = parse_args()
    init_distributed_mode(args)

    torch.backends.cudnn.benchmark = True
    if args.deterministic:
        manual_seed(args.seed + args.local_rank)
        torch.use_deterministic_algorithms(True)

    model_name = args.model.replace('/', '-')
    log_dir = f'{args.output_dir}/{model_name}_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}'
    logger = make_logger(
        f'{args.dataset.lower()}_{model_name}', log_dir,
        rank=args.local_rank
    )

    if args.local_rank == 0:
        logger.info(f'Args: \n{json.dumps(vars(args), indent=4)}')

    model = create_model(
        args.model,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        drop_path_rate=args.drop_path_rate,
        bn_eps=args.bn_eps,
        bn_momentum=args.bn_momentum,
        thumbnail=(args.crop_size < 128),
        pretrained=args.pretrained,
        pth=args.model_path,
        sync_bn=args.sync_bn,
        distributed=args.distributed,
        local_rank=args.local_rank
    )

    model_ema = None
    if args.model_ema:
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        logger.info(f'EMA Decay: {1.0 - alpha}')
        model_ema = ExponentialMovingAverage(model.module if args.distributed else model, device='cuda', decay=1.0 - alpha)

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

    mixupcutmix_fn = None
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(
            RandomMixup(args.num_classes, p=1.0, alpha=args.mixup_alpha)
        )
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(
            RandomCutmix(args.num_classes, p=1.0, alpha=args.cutmix_alpha)
        )
    if mixup_transforms:
        mixupcutmix_fn = T.RandomChoice(mixup_transforms)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    scheduler = create_scheduler(
        args.lr_sched,
        optimizer,
        len(train_loader),
        **(dict(vars(args)))
    )

    if args.local_rank == 0:
        logger.info(f'Model: \n{model}')
        if train_loader.type != "dali":
            logger.info(f'Training: \n{train_loader.dataset.transform}')
        if val_loader.type != "dali":
            logger.info(f'Validation: \n{val_loader.dataset.transform}')
        logger.info(f'Mixup/CutMix: \n{mixupcutmix_fn}')
        logger.info(f'Optimizer: \n{optimizer}')
        logger.info(f'Criterion: {criterion}')
        logger.info(f'Scheduler: {scheduler}')
        logger.info(f'Steps/Epoch: {len(train_loader)}')

    benchmark = Benchmark()
    try:
        for epoch in range(0, args.epochs):
            train(
                train_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                scaler,
                epoch,
                args,
                mixupcutmix_fn,
                model_ema
            )

            validate(val_loader, model, criterion, log_suffix='<VAL> ')
            if model_ema is not None:
                validate(val_loader, model_ema.module, criterion, log_suffix='<EMA> ')

            train_loader.reset()
            val_loader.reset()

            if args.rank == 0 and epoch > (args.epochs - 10):
                model_path = f'{log_dir}/{model_name}_{epoch:0>3}_{time.time()}.pth'
                torch.save(model.module.state_dict(), model_path)
                logger.info(f'Saved: {model_path}!')

                if model_ema is not None:
                    torch.save(model_ema.module.state_dict(), f'{log_dir}/{model_name}_EMA_{epoch:0>3}_{time.time()}.pth')
    except:
        logger.error(traceback.format_exc())

    logger.info(f'Total time: {benchmark.elapsed():>.3f}s')
