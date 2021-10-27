import os
import json
import time
import datetime
import argparse
import torch
import torchvision
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from utils import *
import models

import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy


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
    parser.add_argument('--dropout-rate', type=float, default=0.)
    parser.add_argument('--drop-path-rate', type=float, default=0.)
    parser.add_argument('--crop-size', type=int, default=224)
    parser.add_argument('--val-resize-size', type=int, default=256)
    parser.add_argument('--val-crop-size', type=int, default=224)
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
    parser.add_argument('--min-lr', type=float, default=1e-6)
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
    parser.add_argument('--augment_m', type=float, default=0.4)
    parser.add_argument('--mixup-alpha', type=float, default=0.2, metavar='V',
                        help='beta distribution parameter for mixup sampling. (default: 0.2)')
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
    parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                        help='print frequency. (default: 10)')
    parser.add_argument('--sync_bn', action='store_true',
                        help='use SyncBatchNorm. (default: false)')
    parser.add_argument('--amp', action='store_true',
                        help='mixed precision. (default: false)')
    parser.add_argument('--dali-cpu', action='store_true',
                        help='runs CPU based version of DALI pipeline. (default: false)')
    parser.add_argument('--output-dir', type=str,
                        default=f'logs/{datetime.date.today()}', metavar='DIR')
    return parser.parse_args()


@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True, args=None):
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=is_training,
                                     pad_last_batch=True,
                                     name="Reader")

    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0

    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device,
                                               output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               preallocate_width_hint=preallocate_width_hint,
                                               preallocate_height_hint=preallocate_height_hint,
                                               random_aspect_ratio=[3/4, 4/3],
                                               random_area=[0.08, 1.0],
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_TRIANGULAR)
        if args.augment:
            images = fn.color_twist(images,
                                    device=dali_device,
                                    brightness=fn.random.uniform(
                                        range=[1 - args.augment_m, 1.0 + args.augment_m]),
                                    contrast=fn.random.uniform(
                                        range=[1 - args.augment_m, 1.0 + args.augment_m]),
                                    saturation=fn.random.uniform(
                                        range=[1 - args.augment_m, 1.0 + args.augment_m]),
                                    hue=fn.random.uniform(range=[- args.augment_m * 180, args.augment_m * 180]))  # The hue change in degrees.0~360 or -180~180.
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255, 0.456 *
                                            255, 0.406 * 255],
                                      std=[0.229 * 255, 0.224 *
                                           255, 0.225 * 255],
                                      mirror=mirror)

    labels = labels.gpu()
    return images, labels


def train(train_loader, model, criterion, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze(-1).long()
        original_target = target

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(input)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        acc1, acc5 = accuracy(output, original_target, topk=(1, 5))
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
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    losses = 0

    model.eval()
    end = time.time()
    for i, data in enumerate(val_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze(-1).long()

        with torch.inference_mode():
            output = model(input)
            loss = criterion(output, target)
            losses += loss

        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))
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

    torch.backends.cudnn.benchmark = True
    if args.deterministic:
        manual_seed(args.local_rank)
        torch.use_deterministic_algorithms(True)

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group('nccl')

    logger = make_logger(
        f'imagenet_{args.model}', f'{args.output_dir}/{args.model}', rank=args.local_rank)
    if args.local_rank == 0:
        logger.info(f'Args: \n{json.dumps(vars(args), indent=4)}')

    if args.torch:
        model = torchvision.models.__dict__[
            args.model](pretrained=args.pretrained)
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
    if args.local_rank == 0:
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


    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda()

    pipe = create_dali_pipeline(batch_size=args.batch_size,
                                num_threads=args.workers,
                                device_id=args.local_rank,
                                seed=12 + args.local_rank,
                                data_dir=os.path.join(args.data_dir, 'train'),
                                crop=args.crop_size,
                                size=args.val_resize_size,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=dist.get_world_size(),
                                is_training=True,
                                args=args)
    pipe.build()
    train_loader = DALIClassificationIterator(
        pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

    pipe = create_dali_pipeline(batch_size=args.batch_size,
                                num_threads=args.workers,
                                device_id=args.local_rank,
                                seed=12 + args.local_rank,
                                data_dir=os.path.join(args.data_dir, 'val'),
                                crop=args.val_crop_size,
                                size=args.val_resize_size,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=dist.get_world_size(),
                                is_training=False,
                                args=args)
    pipe.build()
    val_loader = DALIClassificationIterator(
        pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    benchmark = Benchmark()

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
            min_lr=args.min_lr
        )

    if args.local_rank == 0:
        logger.info(f'Optimizer: \n{optimizer}')
        logger.info(f'Criterion: {criterion}')
        logger.info(f'Scheduler: {scheduler}')
        logger.info(f'Steps/Epoch: {len(train_loader)}')

    for epoch in range(0, args.epochs):
        train(train_loader, model, criterion, optimizer,
              scheduler, epoch, args)
        validate(val_loader, model, criterion)
        train_loader.reset()
        val_loader.reset()

        if args.local_rank == 0 and epoch > (args.epochs - 10):
            model_name = f'{args.output_dir}/{args.model}/{args.model}_{epoch:0>3}_{time.time()}.pth'
            torch.save(model.module.state_dict(), model_name)
            logger.info(f'Saved: {model_name}!')
    logger.info(f'Total time: {benchmark.elapsed():>.3f}s')
