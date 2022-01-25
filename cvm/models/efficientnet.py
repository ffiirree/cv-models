from functools import partial
import math
import torch
import torch.nn as nn
from .core import blocks, export, load_from_local_or_url, config
from typing import Any, OrderedDict, List, Type, Union

_BN_EPSILON = 1e-3
# Paper suggests 0.99 momentum
_BN_MOMENTUM = 0.01


def efficientnet_params(model_name):
    """Get efficientnet params based on model name."""
    params_dict = {
        # (width_coefficient, depth_coefficient, resolution, dropout_rate)
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]


@export
class EfficientNet(nn.Module):

    t = [1,   6,  6,  6,  6,   6,   6]  # expand_factor
    c = [32, 16, 24, 40, 80, 112, 192, 320, 1280]  # channels
    n = [1,   2,  2,  3,  3,   4,   1]  # repeats
    k = [3,   3,  3,  3,  3,   3,   3]  # kernel_size
    # k = [3,   3,  5,  3,  5,   5,   3]  # kernel_size

    @blocks.se(partial(nn.SiLU, inplace=True))
    @blocks.nonlinear(partial(nn.SiLU, inplace=True))
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        width_coefficient: float = 1,
        depth_coefficient: float = 1,
        block: Type[Union[blocks.InvertedResidualBlock, blocks.SDInvertedResidualBlock]] = blocks.InvertedResidualBlock,
        se_ratio: float = 0.25,
        dropout_rate: float = 0.2,
        drop_path_rate: float = 0.2,
        dilations: List[int] = None,
        thumbnail: bool = False,
        **kwargs: Any
    ):
        super().__init__()

        dilations = [1] + (dilations or [1, 1, 1, 1])
        assert len(dilations) == 5, ''

        FRONT_S = 1 if thumbnail else 2

        self.s = [1, FRONT_S, 2, 2, 1, 2, 1]  # stride
        stages = [0, 1, 1, 1, 0, 1, 0]   # stages
        ratios = [7/8, 7/8, 6/8, 5/8, 4/8, 4/8, 1/16]

        self.survival_prob = 1 - drop_path_rate
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.dropout_rate = dropout_rate
        self.block = block

        self.n = [self.round_repeats(repeat) for repeat in self.n]
        self.c = [self.round_filters(channels) for channels in self.c]

        self.blocks = sum(self.n)
        self.block_idx = 0

        # first conv3x3
        self.features = nn.Sequential(OrderedDict([
            ('stem', blocks.Stage(
                blocks.Conv2dBlock(in_channels, self.c[0], stride=FRONT_S)
            ))
        ]))

        # blocks
        for i in range(len(self.t)):
            layers = self.make_layers(
                self.c[i],
                self.t[i],
                self.c[i+1],
                self.n[i],
                self.s[i],
                self.k[i],
                se_ratio,
                dilations[len(self.features) + (stages[i] - 1)],
                ratios[i]
            )

            if stages[i]:
                self.features.add_module(f'stage{len(self.features)}', blocks.Stage(layers))
            else:
                self.features[-1].append(layers)

        # last conv1x1
        self.features[-1].append(blocks.Conv2d1x1Block(self.c[-2], self.c[-1]))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.c[-1], num_classes)
        )

    def make_layers(
        self,
        inp: int,
        t: int,
        oup: int,
        n: int,
        stride: int,
        kernel_size: int = 3,
        se_ratio: float = None,
        dilation: int = 1,
        ratio: float = 0.75
    ):
        layers = []
        for i in range(n):
            inp = inp if i == 0 else oup
            stride = stride if i == 0 else 1
            survival_prob = self.survival_prob + (1 - self.survival_prob) * (i + self.block_idx) / self.blocks

            layers.append(
                self.block(
                    inp, oup, t,
                    kernel_size=kernel_size, stride=stride if dilation == 1 else 1,
                    dilation=max(dilation // stride, 1), survival_prob=survival_prob, se_ratio=se_ratio,
                    ratio=ratio if stride == 1 else None
                )
            )

        self.block_idx += n

        return layers

    def round_filters(self, filters: int, divisor: int = 8, min_depth: int = None):
        filters *= self.width_coefficient

        min_depth = min_depth or divisor
        new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(self, repeats: int):
        return int(math.ceil(self.depth_coefficient * repeats))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _effnet(arch, pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    args = efficientnet_params(arch)
    kwargs['dropout_rate'] = kwargs.get('dropout_rate', args[3])
    model = EfficientNet(
        width_coefficient=args[0],
        depth_coefficient=args[1],
        **kwargs
    )

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.1-effnets-weights/efficientnet_b0-002f2cdf.pth')
def efficientnet_b0(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _effnet('efficientnet-b0', pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.1-effnets-weights/sd_efficientnet_b0-7e8d17dc.pth')
def sd_efficientnet_b0(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['block'] = blocks.SDInvertedResidualBlock
    return _effnet('efficientnet-b0', pretrained, pth, progress, **kwargs)


@export
def efficientnet_b1(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _effnet('efficientnet-b1', pretrained, pth, progress, **kwargs)


@export
def efficientnet_b2(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _effnet('efficientnet-b2', pretrained, pth, progress, **kwargs)


@export
def efficientnet_b3(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _effnet('efficientnet-b3', pretrained, pth, progress, **kwargs)


@export
def efficientnet_b4(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _effnet('efficientnet-b4', pretrained, pth, progress, **kwargs)


@export
def efficientnet_b5(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _effnet('efficientnet-b5', pretrained, pth, progress, **kwargs)


@export
def efficientnet_b6(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _effnet('efficientnet-b6', pretrained, pth, progress, **kwargs)


@export
def efficientnet_b7(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _effnet('efficientnet-b7', pretrained, pth, progress, **kwargs)


@export
def efficientnet_b8(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _effnet('efficientnet-b8', pretrained, pth, progress, **kwargs)


@export
def efficientnet_l2(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _effnet('efficientnet-l2', pretrained, pth, progress, **kwargs)
