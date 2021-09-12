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

    cmd = '-m torch.distributed.launch --nproc_per_node=2 train_imagenet.py \
            --data-dir "/datasets/ILSVRC2012" \
            --workers 16 \
            --amp \
            --dali-cpu \
            --lr 0.2 --lr-mode cosine \
            --batch-size 512 \
            --epochs 100 --warmup-epochs 5 \
            --print-freq 250 \
            --momentum 0.9 --wd 0.00001 --no-wd \
            --label-smoothing'

    run_script(cmd, '--augment --model threepathnet_v2_x1_5')
    # run_script(cmd, '--augment --model threepathnet_x2_0')
    run_script(cmd, '--augment --model micronet_d2_0')
    run_script(cmd, '--augment --model micronet_b5_0')

    # run_script(cmd, '--model micronet_b1_0')
    # run_script(cmd, '--model micronet_c1_0')
    # run_script(cmd, '--model micronet_b1_5')
    # run_script(cmd, '--model micronet_c1_5')
    # run_script(cmd, '--augment --model micronet_b2_0')
    # run_script(cmd, '--augment --model micronet_c2_0')
    # run_script(cmd, '--augment --model micronet_b2_5')
    # run_script(cmd, '--augment --model micronet_c2_5')
    
    # run_script(cmd, '--augment --model mobilenet_v2_x0_35')
    # run_script(cmd, '--augment --model mobilenet_v2_x0_5')
    run_script(cmd, '--augment --model shufflenet_v2_x0_5')
    run_script(cmd, '--augment --model mobilenet_v3_small')
    # run_script(cmd, '--augment --model shufflenet_v2_x1_0')
    run_script(cmd, '--augment --model mobilenet_v1_x0_75')
    run_script(cmd, '--augment --model mobilenet_v2_x0_75')
    run_script(cmd, '--augment --model mobilenet_v1_x0_5')
    run_script(cmd, '--augment --model mobilenet_v2_x1_0')
    # run_script(cmd, '--augment --model mobilenet_g5')
    # run_script(cmd, '--augment --model mobilenet_g6')
