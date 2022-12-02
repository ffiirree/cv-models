import json
import argparse
import torch

from tqdm import tqdm

from cvm.utils import *
from cvm.attacks import *


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
    parser.add_argument('--dataset', type=str, default='ImageNet', choices=list_datasets() + ['ImageNet'],
                        help='path to the ImageNet dataset.')
    parser.add_argument('--data-dir', type=str, default='/datasets/ILSVRC2012',
                        help='path to the ImageNet dataset.')
    parser.add_argument('--model', '-m', type=str, default='mobilenet_v1_x1_0', choices=list_models(),
                        help='type of model to use. (default: mobilenet_v1_x1_0)')
    parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                        help='number of label classes')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--model-weights', type=str, default='DEFAULT')
    parser.add_argument('--workers', '-j', type=int, default=8, metavar='N',
                        help='number of data loading workers pre GPU. (default: 3)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='mini-batch size, this is the total batch size of all GPUs. (default: 256)')
    parser.add_argument('--crop-size', type=int, default=224)
    parser.add_argument('--resize-size', type=int, default=232)
    parser.add_argument('--dali', action='store_true', help='use nvidia dali.')
    parser.add_argument('--dali-cpu', action='store_true',
                        help='runs CPU based version of DALI pipeline. (default: false)')
    parser.add_argument('--method', type=str, default='PGD', choices=['FGSM', 'PGD'])
    parser.add_argument('--attack-eps', type=float, default=4/255, metavar='E')
    parser.add_argument('--attack-steps', type=int, default=2, metavar='N')
    parser.add_argument('--attack-alpha', type=float, default=2/255, metavar='A')
    parser.add_argument('--attack-target', type=int, default=-1, metavar='T')
    return parser.parse_args()


if __name__ == '__main__':
    assert torch.cuda.is_available(), 'CUDA IS NOT AVAILABLE!!'
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    init_distributed_mode(args)

    if args.local_rank == 0:
        print(json.dumps(vars(args), indent=4))

    model = create_model(
        args.model,
        pretrained=True,
        thumbnail=(args.crop_size < 128),
        pth=args.model_path,
        weights=args.model_weights,
        distributed=args.distributed,
        local_rank=args.local_rank,
        num_classes=args.num_classes
    )

    val_loader = create_loader(
        args.dataset,
        root=args.data_dir,
        is_training=False,
        batch_size=args.batch_size,
        val_resize_size=args.resize_size,
        val_crop_size=args.crop_size,
        crop_size=args.crop_size,
        workers=args.workers,
        dali=args.dali,
        dali_cpu=args.dali_cpu,
        distributed=args.distributed,
        local_rank=args.local_rank
    )

    if args.local_rank == 0:
        if val_loader.type != "dali":
            print(f'Validation: \n{val_loader.dataset.transform}')

    attacker = None
    if args.method == 'FGSM':
        attacker = FGSM(model, args.attack_eps)
    elif args.method == 'PGD':
        attacker = PGD(model, args.attack_eps, args.attack_steps, args.attack_alpha)
    else:
        raise ValueError(f'Invalid attacker: {args.method}.')

    attacker.set_nomarlized(get_dataset_mean(args.dataset), get_dataset_std(args.dataset))

    if args.local_rank == 0:
        print(f'Attacker: {attacker}')

    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    for (images, target) in tqdm(val_loader, desc='validating', unit='batch'):

        if args.attack_target >= 0:
            target.fill_(args.attack_target)

        images = attacker.perturb(images, target, args.attack_target >= 0)

        with torch.inference_mode():
            output = model(images)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

    acc = f'\n -- top1={top1.avg:6.3f}, top5={top5.avg:6.3f}\n'
    if args.local_rank == 0:
        print(acc)
