import torch
import torch.nn as nn
from .core import *

__all__ = ['ResNetS', 'ResNetM', 'ResNetL']

# 20
class ResNetS(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.filters = filters

        self.features = nn.Sequential(
            Conv2dBlock(in_channels, filters, 7, stride=2, padding=3),

            ResBasicBlock(filters, filters),
            ResBasicBlock(filters, filters * 2),

            ResBasicBlock(filters * 2, filters * 2),
            ResBasicBlock(filters * 2, filters * 4, stride=2),

            ResBasicBlock(filters * 4, filters * 4),
            ResBasicBlock(filters * 4, filters * 4),
            ResBasicBlock(filters * 4, filters * 8, stride=2),

            ResBasicBlock(filters * 8, filters * 8),
            ResBasicBlock(filters * 8, filters * 8),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 8, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 34
class ResNetM(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.filters = filters

        self.features = nn.Sequential(
            Conv2dBlock(in_channels, filters, 7, stride=2, padding=3),

            ResBasicBlock(filters, filters),
            ResBasicBlock(filters, filters),
            ResBasicBlock(filters, filters * 2),

            ResBasicBlock(filters * 2, filters * 2),
            ResBasicBlock(filters * 2, filters * 2),
            ResBasicBlock(filters * 2, filters * 2),
            ResBasicBlock(filters * 2, filters * 4, stride=2),

            ResBasicBlock(filters * 4, filters * 4),
            ResBasicBlock(filters * 4, filters * 4),
            ResBasicBlock(filters * 4, filters * 4),
            ResBasicBlock(filters * 4, filters * 4),
            ResBasicBlock(filters * 4, filters * 4),
            ResBasicBlock(filters * 4, filters * 8, stride=2),

            ResBasicBlock(filters * 8, filters * 8),
            ResBasicBlock(filters * 8, filters * 8),
            ResBasicBlock(filters * 8, filters * 8),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 8, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# 50
class ResNetL(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.filters = filters
        self.layers = [3, 4, 6, 3]

        self.inp = filters
        
        self.features = nn.Sequential(
            Conv2dBlock(in_channels, filters, 7, stride=2, padding=3),
            
            *self.make_layers(filters, self.layers[0]),
            *self.make_layers(filters * 2, self.layers[1]),
            *self.make_layers(filters * 4, self.layers[2]),
            *self.make_layers(filters * 8, self.layers[3]),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 8 * Bottleneck.expansion, num_classes)
        
    def make_layers(self, channels, n):
        layers = []
        
        for _ in range(n):
            layers.append(Bottleneck(self.inp, channels))
            self.inp = channels * Bottleneck.expansion
        return layers

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
