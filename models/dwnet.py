import torch
import torch.nn as nn
from .core import blocks

__all__ = ['DWNet', 'ResDWNet', 'EffDWNet',
           'GroupedDWNet', 'mobilenet_lineardw']


class DwPwBlock(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1
    ):
        self.padding = kernel_size // 2
        super().__init__(
            blocks.DepthwiseConv2d(inp, inp, kernel_size,
                                   stride=stride, padding=self.padding),
            blocks.PointwiseBlock(inp, oup, groups=groups)
        )


class DwPwResBlock(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        groups: int = 1
    ):
        self.padding = kernel_size // 2
        super().__init__()

        self.branch1 = nn.Sequential(
            blocks.DepthwiseConv2d(inp, inp, kernel_size,
                                   stride=1, padding=self.padding),
            blocks.PointwiseBlock(inp, oup, groups=groups)
        )

        self.branch2 = nn.Identity()
        self.combine = blocks.Combine('ADD')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.combine(self.branch1(x), self.branch2(x))
        x = self.relu(x)
        return x


def mobilenet_lineardw(pretrained: bool = False):
    return DWNet()


class DWNet(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters * 1, stride=2),
            DwPwBlock(filters * 1, filters * 2),
            DwPwBlock(filters * 2, filters * 4, stride=2),
            DwPwBlock(filters * 4, filters * 4),
            DwPwBlock(filters * 4, filters * 8, stride=2),
            DwPwBlock(filters * 8, filters * 8),
            DwPwBlock(filters * 8, filters * 16, stride=2),
            DwPwBlock(filters * 16, filters * 16),
            DwPwBlock(filters * 16, filters * 16),
            DwPwBlock(filters * 16, filters * 16),
            DwPwBlock(filters * 16, filters * 16),
            DwPwBlock(filters * 16, filters * 16),
            DwPwBlock(filters * 16, filters * 32, stride=2),
            DwPwBlock(filters * 32, filters * 32),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class EffDwPwBlock(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1
    ):
        self.padding = kernel_size // 2
        super().__init__(
            blocks.PointwiseBlock(inp, inp//2, groups=groups),
            blocks.DepthwiseConv2d(inp // stride, inp // stride, kernel_size,
                                   stride=stride, padding=self.padding),
            blocks.PointwiseBlock(inp // stride, oup, groups=groups)
        )


class EffDWNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters * 1, stride=2),
            DwPwBlock(filters * 1, filters * 2),
            DwPwBlock(filters * 2, filters * 4, stride=2),
            DwPwBlock(filters * 4, filters * 4),
            EffDwPwBlock(filters * 4, filters * 8, stride=2),
            DwPwBlock(filters * 8, filters * 8),
            EffDwPwBlock(filters * 8, filters * 16, stride=2),
            DwPwBlock(filters * 16, filters * 16),
            DwPwBlock(filters * 16, filters * 16),
            DwPwBlock(filters * 16, filters * 16),
            DwPwBlock(filters * 16, filters * 16),
            DwPwBlock(filters * 16, filters * 16),
            EffDwPwBlock(filters * 16, filters * 32, stride=2),
            DwPwBlock(filters * 32, filters * 32),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResDWNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters * 1, stride=2),
            DwPwBlock(filters * 1, filters * 2),
            DwPwBlock(filters * 2, filters * 4, stride=2),
            DwPwResBlock(filters * 4, filters * 4),
            DwPwBlock(filters * 4, filters * 8, stride=2),
            DwPwResBlock(filters * 8, filters * 8),
            DwPwBlock(filters * 8, filters * 16, stride=2),
            DwPwResBlock(filters * 16, filters * 16),
            DwPwResBlock(filters * 16, filters * 16),
            DwPwResBlock(filters * 16, filters * 16),
            DwPwResBlock(filters * 16, filters * 16),
            DwPwResBlock(filters * 16, filters * 16),
            DwPwBlock(filters * 16, filters * 16, stride=2),
            DwPwResBlock(filters * 16, filters * 16),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class GroupedDWNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 30):
        super().__init__()

        assert filters % 6 == 0, ''

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, 32, stride=2),
            DwPwBlock(32 * 1, filters * 2),
            DwPwBlock(filters * 2, filters * 4, stride=2),
            DwPwBlock(filters * 4, filters * 4),
            DwPwBlock(filters * 4, filters * 8, stride=2),
            DwPwBlock(filters * 8, filters * 8),
            DwPwBlock(filters * 8, filters * 16, stride=2, groups=2),
            DwPwBlock(filters * 16, filters * 16, groups=3),
            DwPwBlock(filters * 16, filters * 16, groups=2),
            DwPwBlock(filters * 16, filters * 16, groups=3),
            DwPwBlock(filters * 16, filters * 16, groups=2),
            DwPwBlock(filters * 16, filters * 16, groups=3),
            DwPwBlock(filters * 16, filters * 32, stride=2, groups=2),
            DwPwBlock(filters * 32, filters * 32, groups=3),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
