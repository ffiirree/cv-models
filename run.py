import os
import time


def run_script(script: str, args: str = ''):
    cmd = f'torchrun {script} {args}'
    print(f'\n====\n > {cmd}\n====\n')
    os.system(cmd)
    time.sleep(1)


if __name__ == '__main__':
    cmd = '--standalone --nnodes=1 --nproc_per_node=2 train.py '\
        '--data-dir "/datasets/ILSVRC2012" '\
        '--workers 16 '\
        '--amp '\
        '--dali --dali-cpu '\
        '--lr 0.2 --lr-sched cosine --momentum 0.9 --wd 0.0001 --no-bias-bn-wd '\
        '--batch-size 512 '\
        '--warmup-epochs 5 '\
        '--print-freq 250 '

    run_script(cmd, '--label-smoothing 0.1 --epochs 100 --model resnet18_v1')

