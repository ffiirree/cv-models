import argparse
import torch
import torchvision
from torchinfo import summary
import models

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--model', '-m', type=str)
    parser.add_argument('--torch', action='store_true')

    args = parser.parse_args()

    if args.torch:
        model = torchvision.models.__dict__[args.model]()
    else:
        model = models.__dict__[args.model]()

    summary(
        model,
        input_size=(1, 3, 224, 224),
        col_names=("output_size", "num_params", 'mult_adds')
    )
