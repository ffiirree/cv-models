import torch
import torch.nn as nn
from typing import Optional, Callable

__all__ = ['XNet', 'XNetv2', 'XNetv3', 'XNetv4', 'XNetv5']


class Conv2dBlock(nn.Module):
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
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      bias=False, stride=stride, padding=padding, groups=groups),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.layer(x)


class XNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            Conv2dBlock(in_channels, filters * 1, 7, stride=2, padding=3),
            Conv2dBlock(filters * 1, filters * 1, 3),
            Conv2dBlock(filters * 1, filters * 1, 3),
            Conv2dBlock(filters * 1, filters * 2, 5, stride=2, padding=2),
            Conv2dBlock(filters * 2, filters * 2, 3),
            Conv2dBlock(filters * 2, filters * 2, 3),
            Conv2dBlock(filters * 2, filters * 4, 5, stride=2, padding=2),
            Conv2dBlock(filters * 4, filters * 4, 3),
            Conv2dBlock(filters * 4, filters * 4, 3),
            Conv2dBlock(filters * 4, filters * 8, 5, stride=2, padding=2),
            Conv2dBlock(filters * 8, filters * 8, 3),
            Conv2dBlock(filters * 8, filters * 8, 3),
            Conv2dBlock(filters * 8, filters * 16, 5, stride=2, padding=2),
            Conv2dBlock(filters * 16, filters * 16, 3),
            Conv2dBlock(filters * 16, filters * 16, 3),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class XNetv2(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            Conv2dBlock(in_channels, filters * 1, 7, stride=2, padding=3),
            Conv2dBlock(filters * 1, filters * 1, 3),
            Conv2dBlock(filters * 1, filters * 2, 5, stride=2, padding=2),
            Conv2dBlock(filters * 2, filters * 2, 3),
            Conv2dBlock(filters * 2, filters * 4, 5, stride=2, padding=2),
            Conv2dBlock(filters * 4, filters * 4, 3),
            Conv2dBlock(filters * 4, filters * 8, 5, stride=2, padding=2),
            Conv2dBlock(filters * 8, filters * 8, 3),
            Conv2dBlock(filters * 8, filters * 8, 3),
            Conv2dBlock(filters * 8, filters * 8, 3),
            Conv2dBlock(filters * 8, filters * 12, 5, stride=2, padding=2),
            Conv2dBlock(filters * 12, filters * 12, 3),
            Conv2dBlock(filters * 12, filters * 12, 3),
            Conv2dBlock(filters * 12, filters * 12, 3),
            Conv2dBlock(filters * 12, filters * 16, 3),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class PickLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f'{x.dim()} != 4'
        return torch.cat([x[:, :, i::4, j::4] for i in range(4) for j in range(4)], dim=1)


class XNetv3(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            PickLayer(),
            Conv2dBlock(in_channels * 16, filters * 1, 3),

            Conv2dBlock(filters * 1, filters * 1, 3),
            Conv2dBlock(filters * 1, filters * 2, 3),
            Conv2dBlock(filters * 2, filters * 2, 3),
            Conv2dBlock(filters * 2, filters * 2, 3),
            Conv2dBlock(filters * 2, filters * 4, 5, stride=2, padding=2),
            Conv2dBlock(filters * 4, filters * 4, 3),
            Conv2dBlock(filters * 4, filters * 4, 3),
            Conv2dBlock(filters * 4, filters * 8, 5, stride=2, padding=2),
            Conv2dBlock(filters * 8, filters * 8, 3),
            Conv2dBlock(filters * 8, filters * 8, 3),
            Conv2dBlock(filters * 8, filters * 16, 5, stride=2, padding=2),
            Conv2dBlock(filters * 16, filters * 16, 3),
            Conv2dBlock(filters * 16, filters * 16, 3),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class XNetv4(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            Conv2dBlock(in_channels, filters * 1, 7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Conv2dBlock(filters * 1, filters * 1, 3),

            Conv2dBlock(filters * 1, filters * 1, 3),
            Conv2dBlock(filters * 1, filters * 2, 3),
            Conv2dBlock(filters * 2, filters * 2, 3),
            Conv2dBlock(filters * 2, filters * 2, 3),
            Conv2dBlock(filters * 2, filters * 4, 5, stride=2, padding=2),
            Conv2dBlock(filters * 4, filters * 4, 3),
            Conv2dBlock(filters * 4, filters * 4, 3),
            Conv2dBlock(filters * 4, filters * 8, 5, stride=2, padding=2),
            Conv2dBlock(filters * 8, filters * 8, 3),
            Conv2dBlock(filters * 8, filters * 8, 3),
            Conv2dBlock(filters * 8, filters * 16, 5, stride=2, padding=2),
            Conv2dBlock(filters * 16, filters * 16, 3),
            Conv2dBlock(filters * 16, filters * 16, 3),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# v3 + Group
class XNetv5(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        filters = 24

        self.features = nn.Sequential(
            PickLayer(),
            Conv2dBlock(in_channels * 16, 32, 3),

            Conv2dBlock(32, filters * 1, 3),
            Conv2dBlock(filters * 1, filters * 2, 3, groups=4),
            Conv2dBlock(filters * 2, filters * 2, 3, groups=6),
            Conv2dBlock(filters * 2, filters * 2, 3, groups=4),
            Conv2dBlock(filters * 2, filters * 4, 5,
                        stride=2, padding=2, groups=6),
            Conv2dBlock(filters * 4, filters * 4, 3, groups=4),
            Conv2dBlock(filters * 4, filters * 4, 3, groups=6),
            Conv2dBlock(filters * 4, filters * 8, 5,
                        stride=2, padding=2, groups=4),
            Conv2dBlock(filters * 8, filters * 8, 3, groups=6),
            Conv2dBlock(filters * 8, filters * 8, 3, groups=4),
            Conv2dBlock(filters * 8, filters * 16, 5,
                        stride=2, padding=2, groups=6),
            Conv2dBlock(filters * 16, filters * 16, 3, groups=4),
            Conv2dBlock(filters * 16, filters * 16, 3, groups=6),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x