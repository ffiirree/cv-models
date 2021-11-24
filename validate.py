import argparse
import os
import json
import torch
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
import cvm
from cvm.utils import accuracy, AverageMeter


def parse_args():
    model_names = sorted(name for name in cvm.models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(cvm.models.__dict__[name]))
    model_names += sorted(name for name in torchvision.models.__dict__
                          if name.islower() and not name.startswith("__")
                          and callable(torchvision.models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data-dir', type=str, default='/datasets/ILSVRC2012',
                        help='path to the ImageNet dataset.')
    parser.add_argument('--val-dir', type=str, default='val')
    parser.add_argument('--torch', action='store_true',
                        help='use torchvision models. (default: false)')
    parser.add_argument('--model', type=str, default='muxnet_v2', choices=model_names,
                        help='type of model to use. (default: muxnet_v2)')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pre-trained model. (default: false)')
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--workers', '-j', type=int, default=8, metavar='N',
                        help='number of data loading workers pre GPU. (default: 4)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='mini-batch size, this is the total batch size of all GPUs. (default: 256)')
    parser.add_argument('--crop-size', type=int, default=224)
    parser.add_argument('--resize-size', type=int, default=256)
    return parser.parse_args()


def validate(val_loader, model, args):
    top1 = AverageMeter()
    top5 = AverageMeter()

    model = model.cuda()
    model.eval()
    for images, target in tqdm(val_loader, desc='validating', unit='batch'):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.inference_mode():
            output = model(images)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

    print(f'================\n - top1: {top1.avg:6.3f}\n - top5: {top5.avg:6.3f}\n================')


if __name__ == '__main__':
    assert torch.cuda.is_available(), 'CUDA IS NOT AVAILABLE!!'
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    print(json.dumps(vars(args), indent=4, sort_keys=True))

    if args.torch:
        model = torchvision.models.__dict__[args.model](
            pretrained=args.pretrained
        )
    else:
        model = cvm.models.__dict__[args.model](
            pretrained=args.pretrained,
            pth=args.path
        )

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            os.path.join(args.data_dir, args.val_dir),
            T.Compose([
                T.Resize(args.resize_size),
                T.CenterCrop(args.crop_size),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225)),
            ])
        ),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True
    )

    validate(val_loader, model, args)
