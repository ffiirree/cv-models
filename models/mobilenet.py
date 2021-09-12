import os
import torch
import torch.nn as nn
from .core import blocks

__all__ = ['MobileNet', 'MobileNetLinearDW',
           'mobilenet_lineardw', 'mobilenet_v1_x1_0', 'mobilenet_v1_x0_75',
           'mobilenet_v1_x0_5', 'mobilenet_v1_x0_35']


class MobileBlock(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1
    ):
        super().__init__(
            blocks.DepthwiseBlock(
                inp, inp, kernel_size, stride=stride, padding=padding),
            blocks.PointwiseBlock(inp, oup, groups=groups)
        )


class MobileBlockLinearDW(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1
    ):
        super().__init__(
            blocks.DepthwiseConv2d(
                inp, inp, kernel_size, stride=stride, padding=padding),
            blocks.PointwiseBlock(inp, oup, groups=groups)
        )


def mobilenet_v1_x1_0(pretrained: bool = False, pth: str = None):
    model = MobileNet(depth_multiplier=1.0)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def mobilenet_v1_x0_75(pretrained: bool = False, pth: str = None):
    model = MobileNet(depth_multiplier=0.75)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def mobilenet_v1_x0_5(pretrained: bool = False, pth: str = None):
    model = MobileNet(depth_multiplier=0.5)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def mobilenet_v1_x0_35(pretrained: bool = False, pth: str = None):
    model = MobileNet(depth_multiplier=0.35)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        filters: int = 32,
        depth_multiplier: float = 1.0
    ):
        super().__init__()

        def depth(d): return max(int(d * depth_multiplier), 8)

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, depth(filters), stride=2),
            MobileBlock(depth(filters * 1), depth(filters * 2)),
            MobileBlock(depth(filters * 2), depth(filters * 4), stride=2),
            MobileBlock(depth(filters * 4), depth(filters * 4)),
            MobileBlock(depth(filters * 4), depth(filters * 8), stride=2),
            MobileBlock(depth(filters * 8), depth(filters * 8)),
            MobileBlock(depth(filters * 8), depth(filters * 16), stride=2),
            *[MobileBlock(depth(filters * 16), depth(filters * 16))
              for _ in range(5)],
            MobileBlock(depth(filters * 16), depth(filters * 32), stride=2),
            MobileBlock(depth(filters * 32), depth(filters * 32))
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(depth(filters * 32), num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_lineardw(pretrained: bool = False, pth: str = None):
    model = MobileNetLinearDW()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetLinearDW(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters * 1, stride=2),
            MobileBlockLinearDW(filters * 1, filters * 2),
            MobileBlockLinearDW(filters * 2, filters * 4, stride=2),
            MobileBlockLinearDW(filters * 4, filters * 4),
            MobileBlockLinearDW(filters * 4, filters * 8, stride=2),
            MobileBlockLinearDW(filters * 8, filters * 8),
            MobileBlockLinearDW(filters * 8, filters * 16, stride=2),
            MobileBlockLinearDW(filters * 16, filters * 16),
            MobileBlockLinearDW(filters * 16, filters * 16),
            MobileBlockLinearDW(filters * 16, filters * 16),
            MobileBlockLinearDW(filters * 16, filters * 16),
            MobileBlockLinearDW(filters * 16, filters * 16),
            MobileBlockLinearDW(filters * 16, filters * 32, stride=2),
            MobileBlockLinearDW(filters * 32, filters * 32),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(filters * 32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
