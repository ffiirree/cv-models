import argparse
import torch
from tqdm import tqdm
from torch.profiler.profiler import tensorboard_trace_handler
from cvm.utils import create_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='micronet_b1_0')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--torch', action='store_true')
    args = parser.parse_args()

    model = create_model(args.model, torch=args.torch)
    model.eval()

    images = torch.randn([args.batch_size, 3, 224, 224]).cuda()

    suffix = '_torch' if args.torch else ''
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=2,
            active=2,
            repeat=1
        ),
        profile_memory=True,
        on_trace_ready=tensorboard_trace_handler(
            f'logs/profiles/{args.model}{suffix}'
        ),
        with_stack=True,
        record_shapes=True,
        with_flops=True,
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ]
    ) as prof, tqdm(total=5) as pbar:
        for _ in range(5):
            with torch.cuda.amp.autocast(enabled=args.amp):
                output = model(images)

            prof.step()
            pbar.update()

    print('>>>>>>>> DONE!!!')
