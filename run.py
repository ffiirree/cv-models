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

    # run_script(cmd, '--model mobilenet_lineardw')
    # run_script(cmd, '--model mobilenet_lineardw_v2')
    # run_script(cmd, '--model mobilenet_lineardw_group')
    # run_script(cmd, '--model mobilenet')

    # run_script(cmd, '--model mobilenet_mux_v11')
    # run_script(cmd, '--model mobilenet_mux_v12')
    # run_script(cmd, '--model mobilenet_mux_v11_filter')
    # run_script(cmd, '--model mobilenet_mux_v9')
    # run_script(cmd, '--model mobilenet_mux_v14')
    # run_script(cmd, '--model mobilenet_mux_v13')
    # run_script(cmd, '--model mobilenet_mux_v13')
    # run_script(cmd, '--model mobilenet_mux_v10')

    
    # run_script(cmd, '--model mobilenet_mux_v9')
    # run_script(cmd, '--model mobilenet_mux_v8')
    # run_script(cmd, '--model mobilenet_mux_v7')
    # run_script(cmd, '--model mobilenet_mux_v6')
    # run_script(cmd, '--model mobilenet_mux_v5')

    # run_script(cmd, '--model mobilenet_v2')
    # run_script(cmd, '--torch --model mobilenet_v3_small')
    # run_script(cmd, '--model mobilenet_v3_small_b')
    # run_script(cmd, '--model shufflenet_v2_x1_0')
    # run_script(cmd, '--model mobilenet_v3_small_ob')
    # run_script(cmd, '--model mobilenet_v2_ob')


    # run_script(cmd, '--augment --model mobilenet_g1235')
    # run_script(cmd, '--augment --model mobilenet_g7')
    # run_script(cmd, '--augment --model mobilenet_g12356')
    # run_script(cmd, '--augment --model mobilenet_g5')
    # run_script(cmd, '--augment --model mobilenet_g2')
    # run_script(cmd, '--augment --model mobilenet_g1')

    # run_script(cmd, '--model micronet_a1_0')
    # run_script(cmd, '--model micronet_b1_0')
    run_script(cmd, '--augment --model mobilenet_v3_small')
    run_script(cmd, '--augment --model mobilenet_v3_small_b')
    run_script(cmd, '--augment --model mobilenet_v3_small_c')
    # run_script(cmd, '--model micronet_a1_5')
    # run_script(cmd, '--model micronet_b1_5')
    # run_script(cmd, '--model micronet_c1_5')
