import os
import time
import torch


def run_script(script: str, args: str = ''):
    cmd = f'torchrun {script} {args}'
    print(f'\n====\n > {cmd}\n====\n')
    os.system(cmd)
    time.sleep(1)


if __name__ == '__main__':
    num_devices = torch.cuda.device_count()
    
    # ImageNet
    cmd = f'--standalone --nnodes=1 --nproc_per_node={num_devices} train.py '\
        '--data-dir "/datasets/ILSVRC2012" '\
        '--crop-size 192 --val-resize-size 232 --val-crop-size 224 ' \
        '--workers 32 '\
        '--amp '\
        '--lr 0.2 --lr-sched cosine --momentum 0.9 --wd 0.00001 --no-bias-bn-wd '\
        '--batch-size 512 '\
        '--warmup-epochs 5 '\
        '--print-freq 250 ' \
        '--mixup-alpha 0.8 --cutmix-alpha 1.0 ' \
        '--label-smoothing 0.1 --color-jitter 0.4 --random-erasing 0.25 --epochs 300 '\
        '--model-ema --model-ema-decay 0.9999 --augment rand-m9-mstd0.5'

    # cmd = f'--standalone --nnodes=1 --nproc_per_node={num_devices} train.py '\
    #     '--data-dir "/datasets/ILSVRC2012" '\
    #     '--crop-size 192 --val-resize-size 232 --val-crop-size 224 ' \
    #     '--workers 16 '\
    #     '--amp '\
    #     '--dali --dali-cpu '\
    #     '--lr 0.2 --lr-sched cosine --momentum 0.9 --wd 0.0001 --no-bias-bn-wd '\
    #     '--batch-size 512 '\
    #     '--warmup-epochs 5 '\
    #     '--print-freq 250 ' \
    #     '--label-smoothing 0.1 --epochs 300 '

    # CIFAR10/100
    # cmd = f'--standalone --nnodes=1 --nproc_per_node={num_devices} train.py '\
    #     '--dataset CIFAR100 --data-dir "/datasets/CIFAR100" '\
    #     '--crop-size 32 --val-resize-size 32 --val-crop-size 32 ' \
    #     '--workers 4 '\
    #     '--amp '\
    #     '--lr 0.4 --lr-sched cosine --momentum 0.9 --wd 0.0005 --no-bias-bn-wd '\
    #     '--batch-size 1024 '\
    #     '--warmup-epochs 5 '\
    #     '--print-freq 15 ' \
    #     '--label-smoothing 0.1 --epochs 100 '

    # VOC segmentation
    # cmd = f'--standalone --nnodes=1 --nproc_per_node={num_devices} train_seg.py '\
    #     '--dataset VOCSegmentation --data-dir "/datasets/PASCAL_VOC" '\
    #     '--workers 4 '\
    #     '--amp '\
    #     '--lr 0.01 --lr-sched cosine --momentum 0.9 --wd 0.0001 --no-bias-bn-wd '\
    #     '--batch-size 16 '\
    #     '--print-freq 30 ' \
    #     '--epochs 100 --aux-loss'
    

    run_script(cmd, ' --model efficientnet_b0')
    # run_script(cmd, '--pretrained-backbone --model seg/fcn_regnet_x_400mf')
