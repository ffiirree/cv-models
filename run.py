import os
import time


def run_script(script: str, args: str = ''):
    cmd = f'python {script} {args}'
    print(f'\n====\n > {cmd}\n====\n')
    os.system(cmd)
    time.sleep(1)


if __name__ == '__main__':
    # cmd = '-m torch.distributed.launch --nproc_per_node=2 train_imagenet.py \
    #         --data-dir "/datasets/ILSVRC2012" \
    #         --workers 16 \
    #         --amp \
    #         --lr 1.2 --lr-mode cosine \
    #         --batch-size 1024 \
    #         --epochs 120 --warmup-epochs 5 \
    #         --print-freq 250 \
    #         --momentum 0.9 --wd 0.00003 --no-wd \
    #         --label-smoothing'

    cmd = '-m torch.distributed.launch --nproc_per_node=1 train_imagenet.py \
            --data-dir "/datasets/ILSVRC2012" \
            --workers 16 \
            --amp \
            --dali-cpu \
            --lr 0.2 --lr-mode cosine \
            --batch-size 512 \
            --epochs 100 --warmup-epochs 5 \
            --print-freq 250 \
            --momentum 0.9 --wd 0.00005 --no-wd \
            --label-smoothing'


    run_script(cmd, '--model micronet_se1_0')

