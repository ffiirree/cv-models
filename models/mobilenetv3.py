import os
import torch
import torch.nn as nn
from .core import blocks
from typing import Any

__all__ = ['MobileNetV3Small', 'MobileNetV3Large',
           'mobilenet_v3_small', 'mobilenet_v3_large']

_BN_EPSILON = 1e-3
# Paper suggests 0.99 momentum
_BN_MOMENTUM = 0.01


def mobilenet_v3_small(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = MobileNetV3Small(**kwargs)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetV3Small(nn.Module):

    # @blocks.batchnorm(momentum=_BN_MOMENTUM, eps=_BN_EPSILON)
    @blocks.se(gating_fn=nn.Hardsigmoid)
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        thumbnail: bool  = False
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 2

        self.features = nn.Sequential(
            blocks.Conv2dBlock(
                in_channels, 16, 3, stride=FRONT_S, activation_fn=nn.Hardswish),
            blocks.InvertedResidualBlock(
                16, 16, t=1, kernel_size=3, stride=FRONT_S, se_ratio=0.5, se_ind=True),
            blocks.InvertedResidualBlock(
                16, 24, t=72/16, kernel_size=3, stride=2),
            blocks.InvertedResidualBlock(
                24, 24, t=88/24, kernel_size=3),
            blocks.InvertedResidualBlock(
                24, 40, t=4, kernel_size=5, stride=2, se_ratio=0.25, se_ind=True, activation_fn=nn.Hardswish),
            blocks.InvertedResidualBlock(
                40, 40, t=6, kernel_size=5, stride=1, se_ratio=0.25, se_ind=True, activation_fn=nn.Hardswish),
            blocks.InvertedResidualBlock(
                40, 40, t=6, kernel_size=5, stride=1, se_ratio=0.25, se_ind=True, activation_fn=nn.Hardswish),
            blocks.InvertedResidualBlock(
                40, 48, t=3, kernel_size=5, stride=1, se_ratio=0.25, se_ind=True, activation_fn=nn.Hardswish),
            blocks.InvertedResidualBlock(
                48, 48, t=3, kernel_size=5, stride=1, se_ratio=0.25, se_ind=True, activation_fn=nn.Hardswish),
            blocks.InvertedResidualBlock(
                48, 96, t=6, kernel_size=5, stride=2, se_ratio=0.25, se_ind=True, activation_fn=nn.Hardswish),
            blocks.InvertedResidualBlock(
                96, 96, t=6, kernel_size=5, stride=1, se_ratio=0.25, se_ind=True, activation_fn=nn.Hardswish),
            blocks.InvertedResidualBlock(
                96, 96, t=6, kernel_size=5, stride=1, se_ratio=0.25, se_ind=True, activation_fn=nn.Hardswish),
            blocks.Conv2d1x1Block(
                96, 576, activation_fn=nn.Hardswish),
            # blocks.SEBlock(576, ratio=0.25)
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_v3_large(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = MobileNetV3Large(**kwargs)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetV3Large(nn.Module):

    # @blocks.batchnorm(momentum=_BN_MOMENTUM, eps=_BN_EPSILON)
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        thumbnail: bool  = False
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 2

        self.features = nn.Sequential(
            blocks.Conv2dBlock(
                in_channels,  16, 3, stride=FRONT_S, activation_fn=nn.Hardswish),
            blocks.InvertedResidualBlock(
                16, 16, t=1, kernel_size=3, stride=1),
            blocks.InvertedResidualBlock(
                16, 24, t=4, kernel_size=3, stride=FRONT_S),
            blocks.InvertedResidualBlock(
                24, 24, t=3, kernel_size=3, stride=1),
            blocks.InvertedResidualBlock(
                24, 40, t=3, kernel_size=5, stride=2, se_ratio=0.25, se_ind=True),
            blocks.InvertedResidualBlock(
                40, 40, t=3, kernel_size=5, stride=1, se_ratio=0.25, se_ind=True),
            blocks.InvertedResidualBlock(
                40, 40, t=6, kernel_size=5, stride=1, se_ratio=0.25, se_ind=True),
            blocks.InvertedResidualBlock(
                40, 80, t=6, kernel_size=3, stride=2, activation_fn=nn.Hardswish),
            blocks.InvertedResidualBlock(
                80, 80, t=200/80, kernel_size=3, stride=1, activation_fn=nn.Hardswish),
            blocks.InvertedResidualBlock(
                80, 80, t=184/80, kernel_size=3, stride=1, activation_fn=nn.Hardswish),
            blocks.InvertedResidualBlock(
                80, 80, t=184/80, kernel_size=3, stride=1, activation_fn=nn.Hardswish),
            blocks.InvertedResidualBlock(
                80, 112, t=6, kernel_size=3, stride=1, se_ratio=0.25, se_ind=True, activation_fn=nn.Hardswish),
            blocks.InvertedResidualBlock(
                112, 112, t=6, kernel_size=3, stride=1, se_ratio=0.25, se_ind=True, activation_fn=nn.Hardswish),
            blocks.InvertedResidualBlock(
                112, 160, t=6, kernel_size=5, stride=2, se_ratio=0.25, se_ind=True, activation_fn=nn.Hardswish),
            blocks.InvertedResidualBlock(
                160, 160, t=6, kernel_size=5, stride=1, se_ratio=0.25, se_ind=True, activation_fn=nn.Hardswish),
            blocks.InvertedResidualBlock(
                160, 160, t=6, kernel_size=5, stride=1, se_ratio=0.25, se_ind=True, activation_fn=nn.Hardswish),
            blocks.Conv2d1x1Block(
                160, 960, activation_fn=nn.Hardswish),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
