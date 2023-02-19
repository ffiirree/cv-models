import os
import time
import torch


def run_script(script: str, args: str = ''):
    cmd = f'torchrun --standalone --nnodes=1 --nproc_per_node={torch.cuda.device_count()} {script} {args}'
    print(f'\n====\n > {cmd}\n====\n')
    os.system(cmd)
    time.sleep(1)


if __name__ == '__main__':
    # ImageNet-1K
    imagenet = f'train.py '\
        '--data-dir "/datasets/ILSVRC2012" '\
        '--crop-size 192 --val-resize-size 232 --val-crop-size 224 ' \
        '--workers 16 '\
        '--amp '\
        '--dali --dali-cpu '\
        '--lr 0.2 --lr-sched cosine --momentum 0.9 --wd 0.0001 --no-bias-bn-wd '\
        '--batch-size 512 '\
        '--epochs 100 --warmup-epochs 5 '\
        '--print-freq 250 ' \
        '--label-smoothing 0.1 '
    # '--mixup-alpha 0.8 --cutmix-alpha 1.0 ' \
    # '--color-jitter 0.4 --random-erasing 0.25 '\
    # '--augment rand-m9-mstd0.5 '\
    # '--model-ema --model-ema-decay 0.9999 '

    # ImageNet-398
    tiny_imagenet = f'train.py '\
        '--data-dir "/datasets/TINY_ILSVRC2012" '\
        '--crop-size 176 --val-resize-size 232 --val-crop-size 224 --num-classes 398 ' \
        '--workers 8 '\
        '--amp '\
        '--lr 0.4 --lr-sched cosine --momentum 0.9 --no-bias-bn-wd '\
        '--batch-size 1024 '\
        '--warmup-epochs 5 '\
        '--print-freq 90 ' \
        '--label-smoothing 0.1 '

    mnist = f'train.py '\
        '--dataset MNIST --data-dir "/datasets/MNIST" --in-channels 1 --hflip 0.0 '\
        '--crop-size 28 --val-resize-size 28 --val-crop-size 28 --crop-padding 4 --num-classes 10 ' \
        '--workers 8 '\
        '--amp '\
        '--lr 0.4 --lr-sched cosine --momentum 0.9 --wd 0.001 --no-bias-bn-wd '\
        '--batch-size 2048 --epochs 30 '\
        '--warmup-epochs 3 '\
        '--print-freq 10 ' \
        '--label-smoothing 0.1 '

    # CIFAR10/100
    cifar = f'train.py '\
        '--dataset CIFAR100 --data-dir "/datasets/CIFAR100" '\
        '--crop-size 32 --val-resize-size 32 --val-crop-size 32 ' \
        '--workers 8 '\
        '--amp '\
        '--lr 0.4 --lr-sched cosine --momentum 0.9 --wd 0.0005 --no-bias-bn-wd '\
        '--batch-size 1024 '\
        '--epochs 100 --warmup-epochs 5 '\
        '--print-freq 15 ' \
        '--label-smoothing 0.1 '\
        '--random-erasing 0.25 --dropout-rate 0.25 --augment torch/autoaug-cifar10 '

    # VOC segmentation
    voc = f'train_seg.py '\
        '--dataset VOCSegmentation --data-dir "/datasets/PASCAL_VOC" '\
        '--workers 8 '\
        '--amp '\
        '--lr 0.01 --lr-sched cosine --momentum 0.9 --wd 0.0001 --no-bias-bn-wd '\
        '--batch-size 16 '\
        '--print-freq 30 ' \
        '--epochs 100 --aux-loss'

    run_script(imagenet, '--model mobilenet_v1_x1_0')
    # run_script(voc, '--pretrained-backbone --model seg/fcn_regnet_x_400mf')
