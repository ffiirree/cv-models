#!/usr/bin/python3
import argparse
import torch
import torchvision

import models
from utils import *
from models.core import blocks
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count_table

input = torch.randn(1, 3, 224, 224)


def print_model(model, table: bool = False):
    flops = FlopCountAnalysis(model, input)

    print(flop_count_str(flops) if not table else flop_count_table(flops))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--model', '-m', type=str)
    parser.add_argument('--torch', action='store_true')
    parser.add_argument('--table', action='store_true')
    parser.add_argument('--models', action='store_true')

    args = parser.parse_args()

    if args.torch:
        if args.models:
            print(sorted(name for name in torchvision.models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(torchvision.models.__dict__[name])))
        else:
            print_model(torchvision.models.__dict__[args.model](), args.table)
    else:
        if args.models:
            print(sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name])))
        else:
            print_model(models.__dict__[args.model](), args.table)