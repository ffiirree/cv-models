import os
import torch
import torch.nn as nn
from .core import blocks
from typing import Any

__all__ = ['GoogLeNet', 'googlenet', 'inception_v1']


class InceptionBlock(nn.Module):
    def __init__(
        self,
        inp,
        oup1x1: int,
        oup3x3_r: int,
        oup3x3: int,
        oup5x5_r: int,
        oup5x5: int,
        pool_proj: int
    ):
        super().__init__()

        self.branch1 = blocks.Conv2d1x1Block(inp, oup1x1)

        self.branch2 = nn.Sequential(
            blocks.Conv2d1x1Block(inp, oup3x3_r),
            blocks.Conv2dBlock(oup3x3_r, oup3x3)
        )
        self.branch3 = nn.Sequential(
            blocks.Conv2d1x1Block(inp, oup5x5_r),
            blocks.Conv2dBlock(oup5x5_r, oup5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            blocks.Conv2d1x1Block(inp, pool_proj)
        )

        self.combine = blocks.Combine('CONCAT')

    def forward(self, x):
        x = self.combine([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ])

        return x


class InceptionAux(nn.Sequential):
    def __init__(self, inp, oup):
        super().__init__(
            nn.AdaptiveAvgPool2d((4, 4)),
            blocks.Conv2d1x1Block(inp, 128),
            nn.Flatten(1),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(1024, oup)
        )


def googlenet(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = GoogLeNet(**kwargs)

    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


inception_v1 = googlenet


class GoogLeNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        small_input: bool = False
    ):
        super().__init__()

        FRONT_S = 1 if small_input else 2

        self.conv1 = blocks.Conv2dBlock(
            in_channels, 64, 7, stride=FRONT_S, padding=3)
        self.maxpool1 = nn.Identity() if small_input else nn.MaxPool2d(3, 2, ceil_mode=True)

        self.conv2 = blocks.Conv2d1x1Block(64, 64)
        self.conv3 = blocks.Conv2dBlock(64, 192, 3, padding=1)

        self.maxpool2 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.inception_3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)

        self.maxpool3 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.inception_4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)

        self.maxpool4 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.inception_5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        self.classifiar = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )

        self.aux1 = InceptionAux(512, num_classes)
        self.aux2 = InceptionAux(528, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.conv3(x)

        x = self.maxpool2(x)

        x = self.inception_3a(x)
        x = self.inception_3b(x)

        x = self.maxpool3(x)

        x = self.inception_4a(x)
        aux1 = self.aux1(x) if self.training else None 
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        aux2 = self.aux2(x) if self.training else None 
        x = self.inception_4e(x)

        x = self.maxpool4(x)

        x = self.inception_5a(x)
        x = self.inception_5b(x)

        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifiar(x)

        if self.training:
            return x, aux1, aux2
        else:
            return x
