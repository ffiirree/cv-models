import os
import torch
import torch.nn as nn
from .core import blocks

__all__ = ['MobileNet', 'mobilenet', 'MobileNetLinearDW',
           'mobilenet_lineardw', 'MobileNetLinearDWGroup', 'mobilenet_lineardw_group',
           'mobilenet_lineardw_v2']


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


def mobilenet(pretrained: bool = False, pth: str = None):
    model = MobileNet()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        filters: int = 32
    ):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters * 1, stride=2),
            MobileBlock(filters * 1, filters * 2),
            MobileBlock(filters * 2, filters * 4, stride=2),
            MobileBlock(filters * 4, filters * 4),
            MobileBlock(filters * 4, filters * 8, stride=2),
            MobileBlock(filters * 8, filters * 8),
            MobileBlock(filters * 8, filters * 16, stride=2),
            *[MobileBlock(filters * 16, filters * 16) for _ in range(5)],
            MobileBlock(filters * 16, filters * 32, stride=2),
            MobileBlock(filters * 32, filters * 32)
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(filters * 32, num_classes)
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


class MobileBlockLinearDWv2(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1
    ):
        super().__init__()

        self.inp = inp

        self.dw = blocks.DepthwiseConv2d(inp, inp, kernel_size, stride=stride)
        self.pw1 = blocks.PointwiseBlock(3 * (inp // 4), oup // 2)
        self.pw2 = blocks.PointwiseBlock(3 * (inp // 4), oup // 2)

    def forward(self, x):
        x = self.dw(x)
        x1, _ = torch.split(x, [3 * (self.inp // 4), (self.inp // 4)], dim=1)
        _, x2 = torch.split(x, [(self.inp // 4), 3 * (self.inp // 4)], dim=1)

        x1 = self.pw1(x1)
        x2 = self.pw2(x2)

        return torch.cat([x1, x2], dim=1)


def mobilenet_lineardw_v2(pretrained: bool = False, pth: str = None):
    model = MobileNetLinearDWv2()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetLinearDWv2(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters * 1, stride=2),
            MobileBlockLinearDWv2(filters * 1, filters * 2),
            MobileBlockLinearDWv2(filters * 2, filters * 4, stride=2),
            MobileBlockLinearDWv2(filters * 4, filters * 4),
            MobileBlockLinearDWv2(filters * 4, filters * 8, stride=2),
            MobileBlockLinearDWv2(filters * 8, filters * 8),
            MobileBlockLinearDWv2(filters * 8, filters * 16, stride=2),
            MobileBlockLinearDWv2(filters * 16, filters * 16),
            MobileBlockLinearDWv2(filters * 16, filters * 16),
            MobileBlockLinearDWv2(filters * 16, filters * 16),
            MobileBlockLinearDWv2(filters * 16, filters * 16),
            MobileBlockLinearDWv2(filters * 16, filters * 16),
            MobileBlockLinearDWv2(filters * 16, filters * 32, stride=2),
            MobileBlockLinearDWv2(filters * 32, filters * 32),
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


def mobilenet_lineardw_group(pretrained: bool = False, pth: str = None):
    model = MobileNetLinearDWGroup()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetLinearDWGroup(nn.Module):
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

        )

        self.group1 = nn.Sequential(
            MobileBlockLinearDW(filters * 8, filters * 8),
            MobileBlockLinearDW(filters * 8, filters * 16, stride=2),
            MobileBlockLinearDW(filters * 16, filters * 16)
        )

        self.group2 = nn.Sequential(
            MobileBlockLinearDW(filters * 8, filters * 8),
            MobileBlockLinearDW(filters * 8, filters * 16, stride=2),
            MobileBlockLinearDW(filters * 16, num_classes)
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.conv2 = blocks.DepthwiseConv2d(
            num_classes, num_classes, 7, padding=0)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(filters * 16, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.group1(x1)
        x2 = self.group2(x2)

        x1 = self.avg(x1)
        x2 = self.conv2(x2)

        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)

        x1 = self.classifier(x1)

        return x1 + x2
