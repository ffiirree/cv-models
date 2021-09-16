import os
import torch
import torch.nn as nn
from .core import blocks
from typing import Any

__all__ = ['SqueezeNet', 'squeezenet']


def squeezenet(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = SqueezeNet(**kwargs)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class FireBlock(nn.Module):
    def __init__(self, inp, oup):
        super().__init__()

        planes = oup // 8

        self.squeeze = blocks.Conv2d1x1(inp, planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.expand1x1 = blocks.Conv2d1x1(planes, oup // 2, bias=True)
        self.expand3x3 = blocks.Conv2d3x3(planes, oup // 2, bias=True)
        self.combine = blocks.Combine('CONCAT')
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.relu1(x)
        x = self.combine([self.expand1x1(x), self.expand3x3(x)])
        x = self.relu2(x)
        return x


class SqueezeNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        small_input: bool = False
    ):
        super().__init__()

        FRONT_S = 1 if small_input else 2
        maxpool = nn.Identity() if small_input else nn.MaxPool2d(3, 2, ceil_mode=True)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 96, 7, stride=FRONT_S),
            maxpool,

            FireBlock(96, 128),
            FireBlock(128, 128),
            FireBlock(128, 256),

            nn.MaxPool2d(3, stride=2, ceil_mode=True),

            FireBlock(256, 256),
            FireBlock(256, 384),
            FireBlock(384, 384),
            FireBlock(384, 512),

            nn.MaxPool2d(3, stride=2, ceil_mode=True),

            FireBlock(512, 512)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            blocks.Conv2d1x1(512, num_classes, bias=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
