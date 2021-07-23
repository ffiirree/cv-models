import os
import time

def run_script(script: str, args: str = ''):
    cmd = f'python {script} {args}'
    print(f'\n====\n > {cmd}\n====\n')
    os.system(cmd)
    time.sleep(1)


if __name__ == '__main__':
    cmd = '-m torch.distributed.launch --nproc_per_node=2 \
        train_imagenet.py --data-dir "/datasets/ILSVRC2012" \
            --workers 16 \
                --dali-cpu \
                    --amp \
                        --lr 0.2 --lr-mode cosine --momentum 0.9 --warmup-epochs 5\
                            --batch-size 512 \
                                --mixup --mixup-alpha 0.2 \
                                    --epochs 100'


    # run_script(cmd, '--model mobilenet_lineardw')
    # run_script(cmd, '--model muxnet_v2')
    run_script(cmd, '--model mobilenet')
    run_script(cmd, '--model pwgroupnet')
