import torch
import torch.nn as nn
from .core import blocks

__all__ = ['MobileNet']


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
            MobileBlock(filters * 32, filters * 32),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
