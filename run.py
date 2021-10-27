import os
import time


def run_script(script: str, args: str = ''):
    cmd = f'torchrun {script} {args}'
    print(f'\n====\n > {cmd}\n====\n')
    os.system(cmd)
    time.sleep(1)


if __name__ == '__main__':
    cmd_v1 = '--standalone --nnodes=1 --nproc_per_node=1 train_imagenet.py '\
        '--data-dir "/datasets/ILSVRC2012" '\
        '--workers 32 '\
        '--amp '\
        '--dali-cpu '\
        '--lr 0.2 --lr-mode cosine --min-lr 0.0 --momentum 0.9 --wd 0.0001 --no-wd '\
        '--batch-size 512 '\
        '--epochs 100 --warmup-epochs 5 '\
        '--print-freq 250 '\
        '--label-smoothing 0.1'

    cmd_v2 = '--standalone --nnodes=1 --nproc_per_node=1 train_imagenet_torchvision.py '\
        '--data-dir "/datasets/ILSVRC2012" '\
        '--workers 32 '\
        '--amp '\
        '--lr 0.2 --lr-mode cosine --momentum 0.9 --wd 0.02 --no-wd '\
        '--batch-size 512 '\
        '--epochs 100 --warmup-epochs 5 '\
        '--print-freq 250'

    run_script(cmd_v1, '--model micronet_b1_5')
    run_script(
        cmd_v2,
        '--crop-size 160 --val-resize-size 235 --val-crop-size 224 '
        '--mixup-alpha 0.1 --cutmix-alpha 1.0 '
        '--augment randaugment --randaugment_n 2 --randaugment_m 6 '
        '--model resnet50_v1'
    )
