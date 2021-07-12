import os
import time

def run_script(script:str, args:str=''):
    cmd = f'python {script} {args}'
    print(f'\n====\n > {cmd}\n====\n')
    os.system(cmd)
    time.sleep(1)

if __name__ == '__main__':
    cmd = '-m torch.distributed.launch --nproc_per_node=2 imagenet.py "/datasets/ILSVRC2012" --workers 8 --dali_cpu --amp --lr 0.2 --batch-size 512 --epochs 60'
    # run_script(cmd, '-a XNetv3')
    # run_script(cmd, '-a DWNetv2')
    # run_script(cmd, '-a MicroNet')
    # run_script(cmd, '-a XNetv2')
    # run_script(cmd, '-a XNet')
    # run_script(cmd, '-a XNetv4')
    # run_script(cmd, '-a XNetv5')
    # run_script(cmd, '-a DWNetv3')
    # run_script(cmd, '-a MUXNet --filters 64')
    run_script(cmd, '-a MUXNetv3')
    run_script(cmd, '-a MUXNetv2')
    run_script(cmd, '-a MUXNetv1')