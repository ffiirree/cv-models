import math
import torch
import torch.nn as nn
from .core import blocks

__all__ = ['MobileNetv3Small', 'MobileNetv3Large',
           'mobilenet_v3_small', 'mobilenet_v3_large']

# Paper suggests 0.99 momentum
_BN_MOMENTUM = 0.01


def mobilenet_v3_small():
    return MobileNetv3Small()


def mobilenet_v3_large():
    return MobileNetv3Large()


class MobileNetv3Small(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000
    ):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(
                in_channels,  16, 3, stride=2, activation_layer=nn.Hardswish, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                16, 16, t=1, kernel_size=3, stride=2, se_ratio=0.5, se_ind=True, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                16, 24, t=72/16, kernel_size=3, stride=2, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                24, 24, t=88/24, kernel_size=3, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                24, 40, t=4, kernel_size=5, stride=2, se_ratio=0.25, se_ind=True, activation_layer=nn.Hardswish, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                40, 40, t=6, kernel_size=5, stride=1, se_ratio=0.25, se_ind=True, activation_layer=nn.Hardswish, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                40, 40, t=6, kernel_size=5, stride=1, se_ratio=0.25, se_ind=True, activation_layer=nn.Hardswish, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                40, 48, t=3, kernel_size=5, stride=1, se_ratio=0.25, se_ind=True, activation_layer=nn.Hardswish, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                48, 48, t=3, kernel_size=5, stride=1, se_ratio=0.25, se_ind=True, activation_layer=nn.Hardswish, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                48, 96, t=6, kernel_size=5, stride=2, se_ratio=0.25, se_ind=True, activation_layer=nn.Hardswish, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                96, 96, t=6, kernel_size=5, stride=1, se_ratio=0.25, se_ind=True, activation_layer=nn.Hardswish, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                96, 96, t=6, kernel_size=5, stride=1, se_ratio=0.25, se_ind=True, activation_layer=nn.Hardswish, bn_momentum=_BN_MOMENTUM),
            blocks.Conv2d1x1Block(
                96, 576, activation_layer=nn.Hardswish, bn_momentum=_BN_MOMENTUM),
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


class MobileNetv3Large(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000
    ):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(
                in_channels,  16, 3, stride=2, activation_layer=nn.Hardswish, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                16, 16, t=1, kernel_size=3, stride=1, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                16, 24, t=4, kernel_size=3, stride=2, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                24, 24, t=3, kernel_size=3, stride=1, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                24, 40, t=3, kernel_size=5, stride=2, se_ratio=0.25, se_ind=True, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                40, 40, t=3, kernel_size=5, stride=1, se_ratio=0.25, se_ind=True, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                40, 40, t=6, kernel_size=5, stride=1, se_ratio=0.25, se_ind=True, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                40, 80, t=6, kernel_size=3, stride=2, activation_layer=nn.Hardswish, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                80, 80, t=200/80, kernel_size=3, stride=1, activation_layer=nn.Hardswish, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                80, 80, t=184/80, kernel_size=3, stride=1, activation_layer=nn.Hardswish, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                80, 80, t=184/80, kernel_size=3, stride=1, activation_layer=nn.Hardswish, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                80, 112, t=6, kernel_size=3, stride=1, se_ratio=0.25, se_ind=True, activation_layer=nn.Hardswish, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                112, 112, t=6, kernel_size=3, stride=1, se_ratio=0.25, se_ind=True, activation_layer=nn.Hardswish, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                112, 160, t=6, kernel_size=5, stride=2, se_ratio=0.25, se_ind=True, activation_layer=nn.Hardswish, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                160, 160, t=6, kernel_size=5, stride=1, se_ratio=0.25, se_ind=True, activation_layer=nn.Hardswish, bn_momentum=_BN_MOMENTUM),
            blocks.InvertedResidualBlock(
                160, 160, t=6, kernel_size=5, stride=1, se_ratio=0.25, se_ind=True, activation_layer=nn.Hardswish, bn_momentum=_BN_MOMENTUM),
            blocks.Conv2d1x1Block(
                160, 960, activation_layer=nn.Hardswish, bn_momentum=_BN_MOMENTUM),
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
