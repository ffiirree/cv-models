import os
import json
import time
import datetime
import argparse
import torch
import torch.nn as nn
import torchvision

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
    parser.add_argument('--crop-size', type=int, default=320)
    parser.add_argument('--crop-padding', type=int, default=4, metavar='S')
    parser.add_argument('--val-resize-size', type=int, nargs='+', default=[384, 480])
    parser.add_argument('--val-crop-size', type=int, default=384)

    # model
    parser.add_argument('--model', type=str, default='resnet18_v1', choices=list_models(),
                        help='type of model to use. (default: resnet18_v1)')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pre-trained model. (default: false)')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--num-classes', type=int, default=21, metavar='N',
                        help='number of label classes')
    parser.add_argument('--bn-eps', type=float, default=None)
    parser.add_argument('--bn-momentum', type=float, default=None)
    parser.add_argument('--aux-loss', action='store_true')

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
    parser.add_argument('--random-erasing', type=float,
                        default=0., metavar='P')
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


def train(train_loader, model, criterion, optimizer, scheduler, scaler, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(images)
            loss = torch.stack([criterion(output, targets) for output in outputs]).sum()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - end)

        end = time.time()

        if i % args.print_freq == 0 and i != 0:
            logger.info(f'#{epoch:>3}[{i:>4}] t={batch_time.avg:>.3f}, '
                        f'lr={optimizer.param_groups[0]["lr"]:>.6f}, '
                        f'l={losses.avg:>.3f}')

            if not os.path.exists('logs/voc'):
                os.makedirs('logs/voc')
            
            output = outputs[0].argmax(dim=1)
            targets[targets==255] = 0

            torchvision.utils.save_image(images[0], f'logs/voc/{i}_image.png', normalize=True)
            torchvision.utils.save_image(output[0].float(), f'logs/voc/{i}_pred.png', normalize=True)
            torchvision.utils.save_image(targets[0].float(), f'logs/voc/{i}_mask.png', normalize=True)


def validate(val_loader, model, criterion):
    losses = AverageMeter()

    model.eval()
    for images, targets in val_loader:
        with torch.inference_mode():
            outputs = model(images)
            loss = criterion(outputs[0], targets)

        losses.update(loss.item(), images.size(0))

    logger.info(f'loss={losses.avg:>.5f}')


if __name__ == '__main__':
    assert torch.cuda.is_available(), 'CUDA IS NOT AVAILABLE!!'

    args = parse_args()
    init_distributed_mode(args)

    torch.backends.cudnn.benchmark = True
    if args.deterministic:
        manual_seed(args.seed + args.local_rank)
        torch.use_deterministic_algorithms(True)

    logger = make_logger(
        f'imagenet_{args.model}', f'{args.output_dir}/{args.model}',
        rank=args.local_rank
    )
    if args.local_rank == 0:
        logger.info(f'Args: \n{json.dumps(vars(args), indent=4)}')

    model = create_model(
        args.model,
        num_classes=args.num_classes,
        aux_loss=args.aux_loss,
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
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    train_loader = create_loader(
        root=args.data_dir,
        is_training=True,
        distributed=True,
        taskname='segmentation',
        **(dict(vars(args)))
    )
    val_loader = create_loader(
        root=args.data_dir,
        is_training=False,
        distributed=True,
        taskname='segmentation',
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
            logger.info(f'Training: \n{train_loader.dataset.transforms}')
            logger.info(f'Validation: \n{val_loader.dataset.transforms}')
        logger.info(f'Optimizer: \n{optimizer}')
        logger.info(f'Criterion: {criterion}')
        logger.info(f'Scheduler: {scheduler}')
        logger.info(f'Steps/Epoch: {len(train_loader)}')

    benchmark = Benchmark()
    for epoch in range(0, args.epochs):
        train(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            scaler,
            epoch,
            args
        )

        validate(val_loader, model, criterion)

        train_loader.reset()
        val_loader.reset()

        if args.rank == 0 and epoch > (args.epochs - 10):
            model_name = f'{args.output_dir}/{args.model}/{args.model}_{epoch:0>3}_{time.time()}.pth'
            torch.save(model.module.state_dict(), model_name)
            logger.info(f'Saved: {model_name}!')
    logger.info(f'Total time: {benchmark.elapsed():>.3f}s')
