import os
import time


def run_script(script: str, args: str = ''):
    cmd = f'python {script} {args}'
    print(f'\n====\n > {cmd}\n====\n')
    os.system(cmd)
    time.sleep(1)


if __name__ == '__main__':
    cmd_v1 = '-m torch.distributed.run --nnodes=1 --nproc_per_node=1 train_imagenet.py \
            --data-dir "/datasets/ILSVRC2012" \
            --workers 32 \
            --amp \
            --dali-cpu \
            --lr 0.1 --lr-mode cosine \
            --batch-size 256 \
            --epochs 100 --warmup-epochs 5 \
            --print-freq 500 \
            --momentum 0.9 --wd 0.0001 --no-wd \
            --label-smoothing'

    cmd_v2 = '-m torch.distributed.run --nnodes=1 --nproc_per_node=1 train_imagenet_torchvision.py \
            --data-dir "/datasets/ILSVRC2012" \
            --workers 64 \
            --amp \
            --lr 0.2 --lr-mode cosine \
            --batch-size 512 \
            --epochs 100 --warmup-epochs 5 \
            --print-freq 250 \
            --momentum 0.9 --wd 0.0001 --no-wd \
            --label-smoothing'

    run_script(cmd_v1, '--model efficientnet_b0')
