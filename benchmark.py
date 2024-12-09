import argparse
import torch
import time
from cvm.utils import create_model


class InferenceBenchmarkRunner():
    def __init__(self, model, input, device='cuda', amp=False) -> None:
        self.model = model
        self.input = input
        self.device = device
        self.amp = amp

        self.model = model.to(self.device)
        self.model.eval()
        self.input = input.to(self.device)

    def timestamp(self, sync=False):
        if sync and self.device == 'cuda':
            torch.cuda.synchronize(device=self.device)

        return time.perf_counter()

    def infer(self):
        start = self.timestamp()
        with torch.amp.autocast(device_type='cuda', enabled=self.amp):
            output = self.model(self.input)
        end = self.timestamp(True)
        return end - start


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--model', '-m', type=str)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    print(args)

    model = create_model(args.model)

    input = torch.randn(args.batch_size, 3, 224, 224)

    runner = InferenceBenchmarkRunner(model, input, args.device, args.amp)

    with torch.no_grad():
        for _ in range(50):
            runner.infer()

        total_step = 0
        run_start = runner.timestamp()
        for i in range(50):
            delta_fwd = runner.infer()
            total_step += delta_fwd

        run_end = runner.timestamp(True)
        run_elapsed = run_end - run_start
        print(f'Inference benchmark: {round(50 / run_elapsed, 2):.2f} batches/s, {round(1000 * total_step / 50, 2)} ms')
