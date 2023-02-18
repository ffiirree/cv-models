import json
import argparse
import torch
from cvm.utils import list_models, create_model
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count_table


def print_model(model, str: bool = False, max_depth: int = 3):
    model.eval()
    flops = FlopCountAnalysis(model, input)

    print(flop_count_str(flops) if str else flop_count_table(flops, max_depth=max_depth))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--model', '-m', type=str)
    parser.add_argument('--str', action='store_true')
    parser.add_argument('--list-models', type=str, default=None)
    parser.add_argument('--in-channels', type=int, default=3)
    parser.add_argument('--num-classes', type=int, default=1000)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--max-depth', type=int, default=3)

    args = parser.parse_args()

    input = torch.randn(1, args.in_channels, args.image_size, args.image_size)

    thumbnail = True if args.image_size < 100 else False

    if args.list_models:
        print(json.dumps(list_models(args.list_models), indent=4))
    else:
        print_model(
            create_model(
                args.model,
                thumbnail=thumbnail,
                in_channels=args.in_channels,
                num_classes=args.num_classes,
                cuda=False,
            ),
            args.str,
            args.max_depth
        )
