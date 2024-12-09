import json
import datetime
import argparse
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from cvm.utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch VAE Training')
    # dataset
    parser.add_argument('--data-dir', type=str, default='/datasets/ILSVRC2012',
                        help='path to the ImageNet dataset.')
    parser.add_argument('--dataset', type=str, default='ImageNet', metavar='NAME',
                        choices=list_datasets() + ['ImageNet'], help='dataset type.')
    parser.add_argument('--dataset-download', action='store_true')
    parser.add_argument('--workers', '-j', type=int, default=4, metavar='N',
                        help='number of data loading workers pre GPU. (default: 4)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='mini-batch size, this is the total batch size of all GPUs. (default: 256)')
    parser.add_argument('--crop-size', type=int, default=224)
    parser.add_argument('--crop-padding', type=int, default=0, metavar='S')

    # model
    parser.add_argument('--input-size', type=int, default=28)
    parser.add_argument('--model', type=str, default='vae/vae', choices=list_models(),
                        help='type of model to use. (default: vae/vae)')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pre-trained model. (default: false)')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                        help='number of label classes')
    parser.add_argument('--bn-eps', type=float, default=None)
    parser.add_argument('--bn-momentum', type=float, default=None)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--dropout-rate', type=float, default=0., metavar='P',
                        help='dropout rate. (default: 0.0)')
    parser.add_argument('--drop-path-rate', type=float, default=0., metavar='P',
                        help='drop path rate. (default: 0.0)')

    # optimizer
    parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'rmsprop', 'adam'],
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

    # learning rate
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate. (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=100,  metavar='N',
                        help='number of total epochs to run. (default: 100)')

    # augmentation | regularization
    parser.add_argument('--hflip', type=float, default=0.0, metavar='P')
    parser.add_argument('--vflip', type=float, default=0.0, metavar='P')
    parser.add_argument('--color-jitter', type=float, default=0., metavar='M')
    parser.add_argument('--random-erasing', type=float,
                        default=0., metavar='P')
    parser.add_argument('--augment', type=str, default=None,
                        choices=['randaugment', 'autoaugment'])
    parser.add_argument('--randaugment-n', type=int, default=2, metavar='N',
                        help='RandAugment n.')
    parser.add_argument('--randaugment-m', type=int, default=10, metavar='M',
                        help='RandAugment m.')

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
    parser.add_argument('--output-dir', type=str,
                        default=f'logs/{datetime.date.today()}', metavar='DIR')
    return parser.parse_args()


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
        image_size=args.crop_size,
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

    optimizer = create_optimizer(args.optim, model, **dict(vars(args)))

    def criterion(recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    train_loader = create_loader(
        root=args.data_dir,
        is_training=True,
        download=args.dataset_download,
        mean=(0.5,),
        std=(0.5,),
        **(dict(vars(args)))
    )

    scaler = torch.amp.GradScaler(enabled=args.amp)

    if args.local_rank == 0:
        logger.info(f'Model: \n{model}')
        logger.info(f'Training: \n{train_loader.dataset.transform}')
        logger.info(f'Optimizer: \n{optimizer}')
        logger.info(f'Criterion: {criterion}')
        logger.info(f'Steps/Epoch: {len(train_loader)}')

    benchmark = Benchmark()
    for epoch in range(0, args.epochs):
        model.train()

        for i, (images, _) in enumerate(train_loader):

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=args.amp):
                output, mu, logvar = model(images)
                loss = criterion(output, images, mu, logvar)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_loader.reset()

        save_image(output.detach(), f'{args.output_dir}/rec.png', normalize=True)

        # sample
        with torch.no_grad():
            noise = torch.randn(64, args.nz).cuda()
            sample = model.module.decoder(noise)
            save_image(sample.detach().reshape(64, 1, 28, 28), f'{args.output_dir}/sample.png')

        logger.info(f'#{epoch:>3}/{args.epochs}] loss={loss.item():.3f}')
    logger.info(f'Total time: {benchmark.elapsed():>.3f}s')
