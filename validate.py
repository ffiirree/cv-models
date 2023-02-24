import argparse
import json
import torch
from tqdm import tqdm
from cvm.utils import accuracy, AverageMeter, create_loader, create_model, list_models, list_datasets
from cvm.data import ImageNet1KRealLabelsEvaluator
from cvm.models.ops.functional import *


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
    parser.add_argument('--dataset', type=str, default='ImageNet', choices=list_datasets() + ['ImageNet'],
                        help='path to the ImageNet dataset.')
    parser.add_argument('--data-dir', type=str, default='/datasets/ILSVRC2012',
                        help='path to the ImageNet dataset.')
    parser.add_argument('--model', '-m', type=str, default='mobilenet_v1_x1_0', choices=list_models(),
                        help='type of model to use. (default: mobilenet_v1_x1_0)')
    parser.add_argument('--real-labels', type=str, default=None)
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--model-weights', type=str, default='DEFAULT')
    parser.add_argument('--workers', '-j', type=int, default=8, metavar='N',
                        help='number of data loading workers pre GPU. (default: 4)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='mini-batch size, this is the total batch size of all GPUs. (default: 256)')
    parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                        help='number of label classes')
    parser.add_argument('--in-channels', type=int, default=3, metavar='N')
    parser.add_argument('--crop-size', type=int, default=224)
    parser.add_argument('--resize-size', type=int, default=232)
    parser.add_argument('--dali', action='store_true', help='use nvidia dali.')
    parser.add_argument('--dali-cpu', action='store_true',
                        help='runs CPU based version of DALI pipeline. (default: false)')
    parser.add_argument('--bandpass', type=int, nargs='+', default=None)
    parser.add_argument('--bandreject', type=int, nargs='+', default=None)
    parser.add_argument('--filter-type', type=str, default="ideal", choices=['ideal', 'gaussian'])
    return parser.parse_args()


def validate(val_loader, model, real_evaluator, args):
    top1 = AverageMeter()
    top5 = AverageMeter()

    mask = get_distance_grid(args.crop_size)

    model.eval()
    for (images, target) in tqdm(val_loader, desc='validating', unit='batch'):
        if args.bandpass is not None:
            assert len(args.bandpass) == 2, '--bandpass : [min, max]'
            if args.filter_type == 'ideal':
                kernel = (mask < args.bandpass[0]) | (mask > args.bandpass[1])
                images = spectral_filter(images, lambda fr: torch.masked_fill(fr, kernel.to(fr.device), 0.0))
            elif args.filter_type == 'gaussian':
                kernel = get_gaussian_bandpass_kernel2d(
                    images.size()[-1],
                    (args.bandpass[0] + args.bandpass[1]) / 2,
                    args.bandpass[1] - args.bandpass[0]
                )
                images = spectral_filter(images, lambda fr: fr * kernel.to(fr.device))

        if args.bandreject is not None:
            assert len(args.bandreject) == 2, '--bandreject : [min, max]'
            if args.filter_type == 'ideal':
                kernel = (mask > args.bandreject[0]) & (mask < args.bandreject[1])
                images = spectral_filter(images, lambda fr: lambda fr: torch.masked_fill(fr, kernel.to(fr.device), 0.0))
            elif args.filter_type == 'gaussian':
                kernel = get_gaussian_bandpass_kernel2d(
                    images.size()[-1],
                    (args.bandpass[0] + args.bandpass[1]) / 2,
                    args.bandpass[1] - args.bandpass[0]
                )
                images = spectral_filter(images, lambda fr: fr * (1.0 - kernel.to(fr.devcie)))

        with torch.inference_mode():
            output = model(images)

        if real_evaluator:
            real_evaluator.put(output)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

    print(
        f' ================\n  - top1: {top1.avg:6.3f}\n  - top5: {top5.avg:6.3f}\n ================'
    )
    if real_evaluator:
        print(
            f'Real Labels: \n ================\n  - top1: {real_evaluator.accuracy[1]:6.3f}\n  - top5: {real_evaluator.accuracy[5]:6.3f}\n ================'
        )


if __name__ == '__main__':
    assert torch.cuda.is_available(), 'CUDA IS NOT AVAILABLE!!'
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    print(json.dumps(vars(args), indent=4))

    assert not (args.real_labels and args.dali), ''

    model = create_model(
        args.model,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        thumbnail=(args.crop_size < 128),
        pretrained=True,
        pth=args.model_path,
        weights=args.model_weights
    )

    val_loader = create_loader(
        args.dataset,
        root=args.data_dir,
        is_training=False,
        batch_size=args.batch_size,
        val_resize_size=args.resize_size,
        val_crop_size=args.crop_size,
        workers=args.workers,
        dali=args.dali,
        dali_cpu=args.dali_cpu
    )

    real_evaluator = ImageNet1KRealLabelsEvaluator(
        val_loader.dataset.samples,
        args.real_labels
    ) if args.real_labels else None

    validate(val_loader, model, real_evaluator, args)
