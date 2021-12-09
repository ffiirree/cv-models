from functools import partial
import torch
import torch.nn as nn
from .core import blocks, export, load_from_local_or_url
from typing import Any, OrderedDict, List

_BN_EPSILON = 1e-3
# Paper suggests 0.99 momentum
_BN_MOMENTUM = 0.01

hs = partial(nn.Hardswish, inplace=True)


@export
def mobilenet_v3_small(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    model = MobileNetV3Small(**kwargs)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
class MobileNetV3Small(nn.Module):
    @blocks.se(gating_fn=nn.Hardsigmoid)
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        dropout_rate: float = 0.2,
        dilations: List[int] = None,
        thumbnail: bool = False,
        **kwargs: Any
    ):
        super().__init__()

        dilations = dilations or [1, 1, 1, 1]
        assert len(dilations) == 4, ''

        strides = [2 if dilations[i] == 1 else 1 for i in range(4)]
        FRONT_S = 1 if thumbnail else 2
        strides[0] = FRONT_S

        self.features = nn.Sequential(OrderedDict([
            ('stem', blocks.Stage(
                blocks.Conv2dBlock(in_channels, 16, 3, stride=FRONT_S, activation_fn=hs)
            )),
            ('stage1', blocks.Stage(
                blocks.InvertedResidualBlock(16, 16, 1, kernel_size=3, stride=strides[0], se_ratio=0.5, se_ind=True)
            )),
            ('stage2', blocks.Stage(
                blocks.InvertedResidualBlock(16, 24, 72/16, kernel_size=3, stride=strides[1], dilation=dilations[0]),
                blocks.InvertedResidualBlock(24, 24, 88/24, kernel_size=3, dilation=dilations[1])
            )),
            ('stage3', blocks.Stage(
                blocks.InvertedResidualBlock(24, 40, 4, kernel_size=5, stride=strides[2], dilation=dilations[1], se_ratio=0.25, se_ind=True, activation_fn=hs),
                blocks.InvertedResidualBlock(40, 40, 6, kernel_size=5, dilation=dilations[2], se_ratio=0.25, se_ind=True, activation_fn=hs),
                blocks.InvertedResidualBlock(40, 40, 6, kernel_size=5, dilation=dilations[2], se_ratio=0.25, se_ind=True, activation_fn=hs),
                blocks.InvertedResidualBlock(40, 48, 3, kernel_size=5, dilation=dilations[2], se_ratio=0.25, se_ind=True, activation_fn=hs),
                blocks.InvertedResidualBlock(48, 48, 3, kernel_size=5, dilation=dilations[2], se_ratio=0.25, se_ind=True, activation_fn=hs)
            )),
            ('stage4', blocks.Stage(
                blocks.InvertedResidualBlock(48, 96, 6, kernel_size=5, stride=strides[3], dilation=dilations[2], se_ratio=0.25, se_ind=True, activation_fn=hs),
                blocks.InvertedResidualBlock(96, 96, 6, kernel_size=5, dilation=dilations[3], se_ratio=0.25, se_ind=True, activation_fn=hs),
                blocks.InvertedResidualBlock(96, 96, 6, kernel_size=5, dilation=dilations[3], se_ratio=0.25, se_ind=True, activation_fn=hs),
                blocks.Conv2d1x1Block(96, 576, activation_fn=hs)
            ))
        ]))

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            hs(),
            nn.Dropout(dropout_rate, inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


@export
def mobilenet_v3_large(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    model = MobileNetV3Large(**kwargs)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
class MobileNetV3Large(nn.Module):
    @blocks.se(gating_fn=nn.Hardsigmoid)
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        dropout_rate: float = 0.2,
        dilations: List[int] = None,
        thumbnail: bool = False,
        **kwargs: Any
    ):
        super().__init__()

        dilations = dilations or [1, 1, 1, 1]
        assert len(dilations) == 4, ''

        strides = [2 if dilations[i] == 1 else 1 for i in range(4)]
        FRONT_S = 1 if thumbnail else 2
        strides[0] = FRONT_S

        self.features = nn.Sequential(OrderedDict([
            ('stem', blocks.Stage(
                blocks.Conv2dBlock(in_channels,  16, 3, stride=FRONT_S, activation_fn=hs),
                blocks.InvertedResidualBlock(16, 16, t=1, kernel_size=3, stride=1)
            )),
            ('stage1', blocks.Stage(
                blocks.InvertedResidualBlock(16, 24, t=4, kernel_size=3, stride=strides[0]),
                blocks.InvertedResidualBlock(24, 24, t=3, kernel_size=3, dilation=dilations[0])
            )),
            ('stage2', blocks.Stage(
                blocks.InvertedResidualBlock(24, 40, t=3, kernel_size=5, stride=strides[1], dilation=dilations[0], se_ratio=0.25, se_ind=True),
                blocks.InvertedResidualBlock(40, 40, t=3, kernel_size=5, dilation=dilations[1], se_ratio=0.25, se_ind=True),
                blocks.InvertedResidualBlock(40, 40, t=3, kernel_size=5, dilation=dilations[1], se_ratio=0.25, se_ind=True)
            )),
            ('stage3', blocks.Stage(
                blocks.InvertedResidualBlock(40, 80, t=6, kernel_size=3, stride=strides[2], dilation=dilations[1], activation_fn=hs),
                blocks.InvertedResidualBlock(80, 80, t=200/80, kernel_size=3, dilation=dilations[2], activation_fn=hs),
                blocks.InvertedResidualBlock(80, 80, t=184/80, kernel_size=3, dilation=dilations[2], activation_fn=hs),
                blocks.InvertedResidualBlock(80, 80, t=184/80, kernel_size=3, dilation=dilations[2], activation_fn=hs),
                blocks.InvertedResidualBlock(80, 112, t=6, kernel_size=3, dilation=dilations[2], se_ratio=0.25, se_ind=True, activation_fn=hs),
                blocks.InvertedResidualBlock(112, 112, t=6, kernel_size=3, dilation=dilations[2], se_ratio=0.25, se_ind=True, activation_fn=hs)
            )),
            ('stage4', blocks.Stage(
                blocks.InvertedResidualBlock(112, 160, t=6, kernel_size=5, stride=strides[3], dilation=dilations[2], se_ratio=0.25, se_ind=True, activation_fn=hs),
                blocks.InvertedResidualBlock(160, 160, t=6, kernel_size=5, dilation=dilations[3], se_ratio=0.25, se_ind=True, activation_fn=hs),
                blocks.InvertedResidualBlock(160, 160, t=6, kernel_size=5, dilation=dilations[3], se_ratio=0.25, se_ind=True, activation_fn=hs),
                blocks.Conv2d1x1Block(160, 960, activation_fn=hs)
            ))
        ]))

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            hs(),
            nn.Dropout(dropout_rate, inplace=True),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
