import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear

__all__ = ['SplitNet', 'SplitNetv2', 'SplitNetv3', 'SplitNetv4']

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size:int = 3, stride: int = 1, padding: int = 1, groups: int = 1):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, stride=stride, padding=padding, groups=groups),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.layer(x)

class SplitNet(nn.Module):
    def __init__(self, in_channels:int=3, num_classes:int=1000, filters:int=32):
        super().__init__()

        self.filters = filters

        self.features = nn.Sequential(
            Conv2dBlock(in_channels, filters * 1, 5, 2, padding=2),
            Conv2dBlock(filters * 1, filters * 1, 3),
            Conv2dBlock(filters * 1, filters * 1, 3),
            Conv2dBlock(filters * 1, filters * 2, 5, 2, padding=2),
            Conv2dBlock(filters * 2, filters * 2, 3),
            Conv2dBlock(filters * 2, filters * 2, 3),
            Conv2dBlock(filters * 2, filters * 4, 5, 2, padding=2),
            Conv2dBlock(filters * 4, filters * 4, 3),
            Conv2dBlock(filters * 4, filters * 4, 3),
            Conv2dBlock(filters * 4, filters * 8, 5, 2, padding=2),
            Conv2dBlock(filters * 8, filters * 8, 3),
            Conv2dBlock(filters * 8, filters * 8, 3),
            Conv2dBlock(filters * 8, filters * 16, 5, 2, padding=2),
        )

        self.group1 = nn.Sequential(
            Conv2dBlock(filters * 8, filters * 8, 3),
            Conv2dBlock(filters * 8, filters * 8, 7, padding=0)
        )

        self.group2 =  nn.Sequential(
            Conv2dBlock(filters * 8, filters * 16, 3),
            Conv2dBlock(filters * 16,        1000, 1, padding=0)
        )

        self.fc1 = nn.Linear(self.filters * 8, num_classes)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)

        x1, x2 = torch.split(x, [self.filters * 8, self.filters * 8], dim=1)

        x1 = self.group1(x1)
        x2 = self.group2(x2)
        x2 = self.avg(x2)

        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)

        x1 = self.fc1(x1)
        x1 = x1 + x2

        return x1

class SplitNetv2(nn.Module):
    def __init__(self, in_channels:int=3, num_classes:int=1000, filters:int=32):
        super().__init__()

        self.filters = filters

        self.features = nn.Sequential(
            Conv2dBlock(     in_channels, self.filters * 1, 5, 2, padding=2),
            Conv2dBlock(self.filters * 1, self.filters * 1, 3),
            Conv2dBlock(self.filters * 1, self.filters * 1, 3),
            Conv2dBlock(self.filters * 1, self.filters * 2, 5, 2, padding=2),
            Conv2dBlock(self.filters * 2, self.filters * 2, 3),
            Conv2dBlock(self.filters * 2, self.filters * 2, 3),
            Conv2dBlock(self.filters * 2, self.filters * 4, 5, 2, padding=2),
            Conv2dBlock(self.filters * 4, self.filters * 4, 3),
        )

        self.group1 = nn.Sequential(
            Conv2dBlock(self.filters * 2, self.filters * 2, 3),
            Conv2dBlock(self.filters * 2, self.filters * 4, 5, 2, padding=2),
            Conv2dBlock(self.filters * 4, self.filters * 4, 3),
            Conv2dBlock(self.filters * 4, self.filters * 4, 3),
            Conv2dBlock(self.filters * 4, self.filters * 8, 5, stride=2, padding=2),
            Conv2dBlock(self.filters * 8, self.filters * 8, 3),
            Conv2dBlock(self.filters * 8, self.filters * 8, 7, stride=1, padding=0)
        )

        self.group2 =  nn.Sequential(
            Conv2dBlock(self.filters * 2, self.filters * 2, 3),
            Conv2dBlock(self.filters * 2, self.filters * 4, 5, 2, padding=2),
            Conv2dBlock(self.filters * 4, self.filters * 4, 3),
            Conv2dBlock(self.filters * 4, self.filters * 4, 3),
            Conv2dBlock(self.filters * 4, self.filters * 8, 5, stride=2, padding=2),
            Conv2dBlock(self.filters * 8, self.filters * 8, 3),
            Conv2dBlock(self.filters * 8,      num_classes, 3),
        )

        self.fc1 = nn.Linear(self.filters * 8, num_classes)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)

        x1, x2 = torch.split(x, [self.filters * 2, self.filters * 2], dim=1)

        x1 = self.group1(x1)
        x2 = self.group2(x2)
        x2 = self.avg(x2)

        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)

        x1 = self.fc1(x1)
        x1 = x1 + x2

        return x1

class SplitNetv3(nn.Module):
    def __init__(self, in_channels:int=3, num_classes:int=1000, filters:int=32):
        super().__init__()

        self.filters = filters
        self.num_classes = num_classes

        self.features = nn.Sequential(
            Conv2dBlock(     in_channels, self.filters * 1, 7, stride=2, padding=3),
            Conv2dBlock(self.filters * 1, self.filters * 1, 3),
            Conv2dBlock(self.filters * 1, self.filters * 1, 3),
            Conv2dBlock(self.filters * 1, self.filters * 2, 3, stride=2),
            Conv2dBlock(self.filters * 2, self.filters * 2, 3),
            Conv2dBlock(self.filters * 2, self.filters * 2, 3),
            Conv2dBlock(self.filters * 2, self.filters * 4, 3, stride=2),
            Conv2dBlock(self.filters * 4, self.filters * 4, 3),
        )

        self.group1 = nn.Sequential(
            Conv2dBlock(self.filters * 1, self.filters * 1, 3),
            Conv2dBlock(self.filters * 1, self.filters * 2, 3, stride=2),
            Conv2dBlock(self.filters * 2, self.filters * 2, 3),
            Conv2dBlock(self.filters * 2, self.filters * 2, 3),
            Conv2dBlock(self.filters * 2, self.filters * 4, 3, stride=2),
            Conv2dBlock(self.filters * 4, self.filters * 4, 3),
            Conv2dBlock(self.filters * 4, self.filters * 4, 7, stride=1, padding=0)
        )

        self.group2 =  nn.Sequential(
            Conv2dBlock(self.filters *  3, self.filters *  3, 3),
            Conv2dBlock(self.filters *  3, self.filters *  6, 3, stride=2),
            Conv2dBlock(self.filters *  6, self.filters *  6, 3),
            Conv2dBlock(self.filters *  6, self.filters *  6, 3),
            Conv2dBlock(self.filters *  6, self.filters * 12, 3, stride=2),
            Conv2dBlock(self.filters * 12, self.filters * 12, 3),
            Conv2dBlock(self.filters * 12,  self.num_classes, 3),
        )

        self.fc1 = nn.Linear(self.filters * 4, self.num_classes)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)

        x1, x2 = torch.split(x, [self.filters * 1, self.filters * 3], dim=1)

        x1 = self.group1(x1)
        x2 = self.group2(x2)
        x2 = self.avg(x2)

        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)

        x1 = self.fc1(x1)
        x1 = x1 + x2

        return x1


class BasicGroup(nn.Module):
    def __init__(self, filters: int, num_classes: int):
        super().__init__()

        self.block = nn.Sequential(
            Conv2dBlock(filters * 1, filters * 2, 5, 2, padding=2),
            Conv2dBlock(filters * 2, filters * 2, 3),
            Conv2dBlock(filters * 2, filters * 2, 3),
            Conv2dBlock(filters * 2, filters * 4, 5, stride=2, padding=2),
            Conv2dBlock(filters * 4, filters * 4, 3),
            Conv2dBlock(filters * 4, filters * 4, 7, padding=0),
            nn.Flatten(1),
            nn.Linear(filters * 4, num_classes)
        )

    def forward(self, x):
        return self.block(x)

class MappingGroup(nn.Module):
    def __init__(self, filters: int, num_classes: int):
        super().__init__()

        self.block = nn.Sequential(
            Conv2dBlock(filters * 1, filters * 2, 5, 2, padding=2),
            Conv2dBlock(filters * 2, filters * 2, 3),
            Conv2dBlock(filters * 2, filters * 2, 3),
            Conv2dBlock(filters * 2, filters * 4, 5, stride=2, padding=2),
            Conv2dBlock(filters * 4, filters * 8, 3),
            Conv2dBlock(filters * 8, num_classes, 1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1)
        )

    def forward(self, x):
        return self.block(x)

class SplitNetv4(nn.Module):
    def __init__(self, in_channels:int=3, num_classes:int=1000, filters:int=32):
        super().__init__()

        self.filters = filters

        self.features = nn.Sequential(
            Conv2dBlock(in_channels, filters * 1, 5, 2, padding=2),
            Conv2dBlock(filters * 1, filters * 1, 3),
            Conv2dBlock(filters * 1, filters * 1, 3),
            Conv2dBlock(filters * 1, filters * 2, 5, 2, padding=2),
            Conv2dBlock(filters * 2, filters * 2, 3),
            Conv2dBlock(filters * 2, filters * 2, 3),
            Conv2dBlock(filters * 2, filters * 4, 5, 2, padding=2),
            Conv2dBlock(filters * 4, filters * 4, 3),
            Conv2dBlock(filters * 4, filters * 4, 3),
        )

        self.group1 = BasicGroup(filters, num_classes)
        self.group2 = BasicGroup(filters, num_classes)
        self.group3 = MappingGroup(filters, num_classes)
        self.group4 = MappingGroup(filters, num_classes)

    def forward(self, x):
        x = self.features(x)

        x1, x2, x3, x4 = torch.split(x, self.filters, dim=1)

        x1 = self.group1(x1)
        x2 = self.group2(x2)
        x3 = self.group3(x3)
        x4 = self.group4(x4)

        return x1 + x2 + x3 + x4