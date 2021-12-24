import os
import time


def run_script(script: str, args: str = ''):
    cmd = f'torchrun {script} {args}'
    print(f'\n====\n > {cmd}\n====\n')
    os.system(cmd)
    time.sleep(1)


if __name__ == '__main__':
    cmd = '--standalone --nnodes=1 --nproc_per_node=4 train.py '\
        '--data-dir "/datasets/ILSVRC2012" '\
        '--workers 3 '\
        '--amp '\
        '--dali --dali-cpu '\
        '--lr 0.2 --lr-sched cosine --momentum 0.9 --wd 0.0001 --no-bias-bn-wd '\
        '--batch-size 512 '\
        '--warmup-epochs 5 '\
        '--print-freq 250 ' \
        '--label-smoothing 0.1 --epochs 100 '

    # cmd = '--standalone --nnodes=1 --nproc_per_node=4 train.py '\
    #     '--dataset CIFAR100 --data-dir "/home/zhliangqi/data/datasets/CIFAR100" '\
    #     '--crop-size 32 --val-resize-size 32 --val-crop-size 32 ' \
    #     '--workers 3 '\
    #     '--amp '\
    #     '--lr 0.4 --lr-sched cosine --momentum 0.9 --wd 0.0005 --no-bias-bn-wd '\
    #     '--batch-size 1024 '\
    #     '--warmup-epochs 5 '\
    #     '--print-freq 15 ' \
    #     '--augment autoaugment --label-smoothing 0.1 --epochs 100 '


    # run_script(cmd, '--crop-size 128 --val-resize-size 144 --val-crop-size 128 --model vgnetg_1_0mp')
    run_script(cmd, '--model vgnetg_1_0mp')
    # run_script(cmd, '--model vgnetg_1_5mp')
    # run_script(cmd, '--model vgnetg_2_0mp')
    # run_script(cmd, '--model vgnetg_2_5mp')
    



