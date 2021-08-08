import os
import time


def run_script(script: str, args: str = ''):
    cmd = f'python {script} {args}'
    print(f'\n====\n > {cmd}\n====\n')
    os.system(cmd)
    time.sleep(1)


if __name__ == '__main__':
    cmd = '-m torch.distributed.launch --nproc_per_node=2 train_imagenet.py \
            --data-dir "/datasets/ILSVRC2012" \
            --workers 16 \
            --amp \
            --dali-cpu \
            --lr 0.064 --lr-mode step --lr-decay-epochs 1 --lr-decay 0.954 \
            --batch-size 1024 \
            --epochs 150 --warmup-epochs 5 \
            --print-freq 250 \
            --momentum 0.9 --wd 0.00001 --no-wd \
            --optim rmsprop --rmsprop-decay 0.9 --rmsprop-epsilon 0.0316'

    # run_script(cmd, '--model mobilenet_lineardw')
    # run_script(cmd, '--model mobilenet_lineardw_v2')
    # run_script(cmd, '--model mobilenet_lineardw_group')
    # run_script(cmd, '--model mobilenet')

    # run_script(cmd, '--model mobilenet_v2')
    run_script(cmd, '--torch --model mobilenet_v3_small')
    run_script(cmd, '--model mobilenet_v3_small')
    # run_script(cmd, '--model shufflenet_v2_x1_0')
    # run_script(cmd, '--model efficientnet_b0')
