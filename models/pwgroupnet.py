import torch
import torch.nn as nn
from .core import blocks

__all__ = ['PWGroupNet', 'pwgroupnet']


class MobileBlock(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__(
            blocks.DepthwiseBlock(
                inp, inp, kernel_size=kernel_size, stride=stride, padding=padding),
            blocks.PointwiseBlock(inp, oup)
        )


class PWGroupBlock(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()

        self.dw = blocks.DepthwiseConv2d(
            inp, inp, kernel_size=kernel_size, stride=stride, padding=padding)
        self.pw3 = blocks.PointwiseConv2d(inp, oup // 4, groups=3)
        self.pw2 = blocks.PointwiseConv2d(inp, oup // 4, groups=2)
        self.pw1 = blocks.PointwiseConv2d(inp, oup // 2)
        self.nolinear = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        x = self.dw(x)
        x = torch.cat([self.pw1(x), self.pw2(x), self.pw3(x)], dim=1)
        x = self.nolinear(x)
        return x


def pwgroupnet(pretrained: bool = False):
    return PWGroupNet()


class PWGroupNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        filters: int = 30
    ):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters * 1, stride=2),
            MobileBlock(filters * 1, filters * 2),
            MobileBlock(filters * 2, filters * 4, stride=2),
            MobileBlock(filters * 4, filters * 4),
            MobileBlock(filters * 4, filters * 8, stride=2),
            PWGroupBlock(filters * 8, filters * 8),
            PWGroupBlock(filters * 8, filters * 16, stride=2),
            *[PWGroupBlock(filters * 16, filters * 16) for _ in range(5)],
            PWGroupBlock(filters * 16, filters * 32, stride=2),
            PWGroupBlock(filters * 32, filters * 32),
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
