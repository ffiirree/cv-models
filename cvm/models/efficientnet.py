from functools import partial
import math
import torch
import torch.nn as nn
from .core import blocks, export, load_from_local_or_url
from typing import Any

_BN_EPSILON = 1e-3
# Paper suggests 0.99 momentum
_BN_MOMENTUM = 0.01


@export
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

    t = [1, 6, 6, 6, 6, 6, 6]  # expand_factor
    c = [32, 16, 24, 40, 80, 112, 192, 320, 1280]  # channels
    n = [1, 2, 2, 3, 3, 4, 1]  # repeats
    k = [3, 3, 5, 3, 5, 5, 3]  # kernel_size

    @blocks.se(partial(nn.SiLU, inplace=True))
    @blocks.nonlinear(partial(nn.SiLU, inplace=True))
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        width_coefficient: float = 1,
        depth_coefficient: float = 1,
        dropout_rate: float = 0.2,
        drop_path_rate: float = 0.2,
        thumbnail: bool = False,
        **kwargs: Any
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 2

        self.s = [1, FRONT_S, 2, 2, 1, 2, 1]  # stride

        self.survival_prob = 1 - drop_path_rate
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.dropout_rate = dropout_rate

        self.n = [self.round_repeats(repeat) for repeat in self.n]
        self.c = [self.round_filters(channels) for channels in self.c]

        self.blocks = sum(self.n)
        self.block_idx = 0

        # first conv3x3
        features = [blocks.Conv2dBlock(in_channels, self.c[0], stride=FRONT_S)]

        # blocks
        for i in range(len(self.t)):
            features.append(
                self.make_layers(
                    self.c[i],
                    self.t[i],
                    self.c[i+1],
                    self.n[i],
                    self.s[i],
                    self.k[i],
                    0.25
                )
            )

        # last conv1x1
        features.append(blocks.Conv2d1x1Block(self.c[-2], self.c[-1]))

        self.features = nn.Sequential(*features)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
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
        se_ratio: float = None
    ):
        layers = []
        for i in range(n):
            inp = inp if i == 0 else oup
            stride = stride if i == 0 else 1
            survival_prob = self.survival_prob + \
                (1 - self.survival_prob) * (i + self.block_idx) / self.blocks

            layers.append(
                blocks.InvertedResidualBlock(
                    inp, oup, t,
                    kernel_size=kernel_size, stride=stride,
                    survival_prob=survival_prob, se_ratio=se_ratio
                )
            )

        self.block_idx += n

        return nn.Sequential(*layers)

    def round_filters(self, filters: int, divisor: int = 8, min_depth: int = None):
        filters *= self.width_coefficient

        min_depth = min_depth or divisor
        new_filters = max(min_depth, int(
            filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(self, repeats: int):
        return int(math.ceil(self.depth_coefficient * repeats))

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
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
def efficientnet_b0(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
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
