import torch
import torch.nn as nn
from .core import blocks

__all__ = ['DWNet', 'DWNetv2', 'DWNetv3']


class DWBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                      bias=False, stride=stride, padding=padding, groups=in_channels),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=1, bias=False, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.layer(x)


class DWNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters * 1, stride=2),
            DWBlock(filters * 1, filters * 2),
            DWBlock(filters * 2, filters * 4, stride=2),
            DWBlock(filters * 4, filters * 4),
            DWBlock(filters * 4, filters * 8, stride=2),
            DWBlock(filters * 8, filters * 8),
            DWBlock(filters * 8, filters * 16, stride=2),
            DWBlock(filters * 16, filters * 16),
            DWBlock(filters * 16, filters * 16),
            DWBlock(filters * 16, filters * 16),
            DWBlock(filters * 16, filters * 16),
            DWBlock(filters * 16, filters * 16),
            DWBlock(filters * 16, filters * 32, stride=2),
            DWBlock(filters * 32, filters * 32),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class DWBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                      bias=False, stride=stride, padding=padding, groups=in_channels),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=1, bias=False, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.layer(x)


class DWNetv2(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters * 1, stride=2),
            DWBlock2(filters * 1, filters * 2),
            DWBlock2(filters * 2, filters * 4, stride=2),
            DWBlock2(filters * 4, filters * 4),
            DWBlock2(filters * 4, filters * 8, stride=2),
            DWBlock2(filters * 8, filters * 8),
            DWBlock2(filters * 8, filters * 16, stride=2),
            DWBlock2(filters * 16, filters * 16),
            DWBlock2(filters * 16, filters * 16),
            DWBlock2(filters * 16, filters * 16),
            DWBlock2(filters * 16, filters * 16),
            DWBlock2(filters * 16, filters * 16),
            DWBlock2(filters * 16, filters * 32, stride=2),
            DWBlock2(filters * 32, filters * 32),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class DWBlockv3(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1
    ):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                      bias=False, stride=stride, padding=padding, groups=in_channels),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=1, bias=False, padding=0, groups=groups),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.layer(x)


class DWNetv3(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        filters = 24

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters * 1, stride=2),
            DWBlockv3(filters * 1, filters * 2),
            DWBlockv3(filters * 2, filters * 4, stride=2),
            DWBlockv3(filters * 4, filters * 4),
            DWBlockv3(filters * 4, filters * 8, stride=2),
            DWBlockv3(filters * 8, filters * 8, groups=4),
            DWBlockv3(filters * 8, filters * 16, stride=2, groups=6),
            DWBlockv3(filters * 16, filters * 16, groups=4),
            DWBlockv3(filters * 16, filters * 16, groups=6),
            DWBlockv3(filters * 16, filters * 16, groups=4),
            DWBlockv3(filters * 16, filters * 16, groups=6),
            DWBlockv3(filters * 16, filters * 16, groups=4),
            DWBlockv3(filters * 16, filters * 32, stride=2, groups=6),
            DWBlockv3(filters * 32, filters * 32, groups=4),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
