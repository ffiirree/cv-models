import argparse
import os
import math
import torch
import torchvision
import torch.distributed as dist
import time
from utils import *
import models

from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from thop import profile

def parse_args():
    # model_names = sorted(name for name in torchvision.models.__dict__
    #                  if name.islower() and not name.startswith("__")
    #                  and callable(torchvision.models.__dict__[name]))

    model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='XNet',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: XNet)')
    # parser.add_argument('--pretrained', dest='pretrained', action='store_true',
    #                     help='use pre-trained model')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--local_rank', metavar='RANK', type=int, default=0)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--filters', metavar='FILTERS', type=int, default=32)
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--print-freq', '-p', default=25, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')
    parser.add_argument('--amp',                action='store_true')
    parser.add_argument('--dali_cpu', action='store_true',
                        help='Runs CPU based version of DALI pipeline.')
    parser.add_argument('--output_dir', type=str, default='logs')
    return parser.parse_args()

@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=is_training,
                                     pad_last_batch=True,
                                     name="Reader")
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               random_aspect_ratio=[0.8, 1.25],
                                               random_area=[0.1, 1.0],
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_TRIANGULAR)
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
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)
    labels = labels.gpu()
    return images, labels


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    model.train()
    
    end = time.time()
    for i, data in enumerate(train_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze(-1).long()

        adjust_learning_rate(optimizer, epoch, i, int(math.ceil(train_loader._size / args.batch_size)))

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(input)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(f'#{epoch:>3} [{args.local_rank}:{i:>4}/{len(train_loader)}], '
                        f'lr={optimizer.param_groups[0]["lr"]:>.10f}, '
                        f't={batch_time.val:>.3f}/{batch_time.avg:>.3f}, '
                        f'l={losses.val:>.5f}/{losses.avg:>.5f}, '
                        f't1={top1.val:>6.3f}/{top1.avg:>6.3f}, '
                        f't5={top5.val:>6.3f}/{top5.avg:>6.3f}')

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

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)
            losses += loss

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(f'Validation [{i:>3}/{len(val_loader)}], '
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
        logger.info(f'loss={losses.item() / (len(val_loader) * dist.get_world_size()):>.5f}, '
                    f'top1={top1.item() / dist.get_world_size():>6.3f}, '
                    f'top5={top5.item() / dist.get_world_size():>6.3f}')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    warmup_epochs = 5

    factor = epoch // 10

    if epoch >= 55:
        factor = factor + 1

    lr = args.lr*(0.2**factor)

    """Warmup"""
    if epoch < warmup_epochs:
        lr = lr * float(1 + step + epoch * len_epoch) / (warmup_epochs * len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    assert torch.cuda.is_available(), 'CUDA IS NOT AVAILABLE!!'

    args = parse_args()
    args.batch_size = int(args.batch_size / torch.cuda.device_count())

    logger = make_logger(f'imagenet_{args.arch}', 'logs')
    logger.info(args)

    torch.backends.cudnn.benchmark = True
    if args.deterministic:
        manual_seed(args.local_rank)

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group('nccl')

    model = models.__dict__[args.arch](3, 1000, args.filters)
    # model = torchvision.models.__dict__[args.arch]()
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    model = model.cuda()
    if args.local_rank == 0:
        logger.info(f'Model: \n{model}')

        # macs, params = profile(model, inputs=(torch.randn(1, 3, 224, 224).cuda(),))
        # logger.info(f'MACs: {macs / 1000000000:>.3f}G, Params: {params / 1000000:.2f}M (Store: {params / (256 * 1024):>.2f}MB)')
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    
    pipe = create_dali_pipeline(batch_size=args.batch_size,
                                num_threads=args.workers,
                                device_id=args.local_rank,
                                seed=12 + args.local_rank,
                                data_dir=os.path.join(args.data, 'train'),
                                crop=224,
                                size=256,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=dist.get_world_size(),
                                is_training=True)

    pipe.build()
    train_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)


    pipe = create_dali_pipeline(batch_size=args.batch_size,
                                num_threads=args.workers,
                                device_id=args.local_rank,
                                seed=12 + args.local_rank,
                                data_dir=os.path.join(args.data, 'val'),
                                crop=224,
                                size=256,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=dist.get_world_size(),
                                is_training=False)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
    
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    benchmark = Benchmark()
    if args.evaluate:
        validate(val_loader, model, criterion)
    else:
        for epoch in range(0, args.epochs):
            train(train_loader, model, criterion, optimizer, epoch, args)
            validate(val_loader, model, criterion)
            train_loader.reset()
            val_loader.reset()

            if args.local_rank == 0:
                model_name = f'{args.output_dir}/{args.arch}_{time.time()}_{epoch}.pt'
                torch.save(model.module.state_dict(), model_name)
                logger.info(f'Saved: {model_name}!')
    logger.info(f'Total time: {benchmark.elapsed():>.3f}s')



