import json
import argparse
import torch
from cvm.utils import list_models, create_model
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count_table


def print_model(model, table: bool = False):
    model.eval()
    flops = FlopCountAnalysis(model, input)

    print(flop_count_str(flops) if not table else flop_count_table(flops))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--model', '-m', type=str)
    parser.add_argument('--torch', action='store_true')
    parser.add_argument('--table', action='store_true')
    parser.add_argument('--models', action='store_true')
    parser.add_argument('--num-classes', type=int, default=1000)
    parser.add_argument('--image-size', type=int, default=224)

    args = parser.parse_args()

    input = torch.randn(1, 3, args.image_size, args.image_size)

    thumbnail = True if args.image_size < 100 else False

    if args.models:
        print(json.dumps(list_models(args.torch), indent=4))
    else:
        print_model(
            create_model(
                args.model,
                thumbnail=thumbnail,
                torch=args.torch,
                num_classes=args.num_classes
            ),
            args.table
        )
