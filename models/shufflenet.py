import os
import torch
import torch.nn as nn
from .core import blocks
from typing import Any, OrderedDict, List


__all__ = ['ShuffleNet', 'shufflenet_g1', 'shufflenet_g2',
           'shufflenet_g3', 'shufflenet_g4', 'shufflenet_g8']


class ShuffleAddBlock(nn.Module):
    def __init__(self, channels, g: int = 2):
        super().__init__()

        self.branch1 = nn.Sequential(OrderedDict([
            ('gconv1', blocks.Conv2d1x1Block(channels, channels, groups=g)),
            ('shuffle', blocks.ChannelShuffle(groups=g)),
            ('dwconv', blocks.DepthwiseConv2dBN(channels, channels, 3)),
            ('gconv2', blocks.Conv2d1x1BN(channels, channels, groups=g))
        ]))

        self.branch2 = nn.Identity()
        self.combine = blocks.Combine('ADD')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.combine([self.branch1(x), self.branch2(x)])
        x = self.relu(x)
        return x


class ShuffleCatBlock(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        g: int = 2,
        stride: int = 2,
        apply_first: bool = True
    ):
        super().__init__()

        g_1st = g if apply_first else 1

        self.branch1 = nn.Sequential(OrderedDict([
            ('gconv1', blocks.Conv2d1x1Block(inp, oup, groups=g_1st)),
            ('shuffle', blocks.ChannelShuffle(groups=g)),
            ('dwconv', blocks.DepthwiseConv2dBN(oup, oup, stride=stride)),
            ('gconv2', blocks.Conv2d1x1BN(oup, oup, groups=g))
        ]))

        self.branch2 = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        self.combine = blocks.Combine('CONCAT')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.combine([self.branch1(x), self.branch2(x)])
        x = self.relu(x)
        return x


def shufflenet_g1(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = ShuffleNet(
        repeats=[4, 8, 4],
        channels=[24, 144, 288, 576],
        g=1,
        **kwargs
    )

    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))

    return model


def shufflenet_g2(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = ShuffleNet(
        repeats=[4, 8, 4],
        channels=[24, 200, 400, 800],
        g=2,
        **kwargs
    )
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def shufflenet_g3(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = ShuffleNet(
        repeats=[4, 8, 4],
        channels=[24, 240, 480, 960],
        g=3,
        **kwargs
    )

    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))

    return model


def shufflenet_g4(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = ShuffleNet(
        repeats=[4, 8, 4],
        channels=[24, 272, 544, 1088],
        g=4,
        **kwargs
    )

    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))

    return model


def shufflenet_g8(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = ShuffleNet(
        repeats=[4, 8, 4],
        channels=[24, 384, 768, 1536],
        g=8,
        **kwargs
    )

    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))

    return model


class ShuffleNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        repeats: List[int] = [4, 84, 4],
        channels: List[int] = [],
        g: int = 3,
        thumbnail: bool = False
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 2

        self.conv1 = blocks.Conv2dBlock(in_channels, channels[0], 3, FRONT_S)
        self.down1 = nn.MaxPool2d(kernel_size=3, stride=2)
        if thumbnail:
            self.down1 = nn.Identity()

        self.stage2 = self.make_layers(repeats[0], channels[0], channels[1], g)
        self.stage3 = self.make_layers(repeats[1], channels[1], channels[2], g)
        self.stage4 = self.make_layers(repeats[2], channels[2], channels[3], g)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[3], num_classes)

    @staticmethod
    def make_layers(repeat, inp, oup, g):
        layers = [ShuffleCatBlock(inp, oup - inp, stride=2, g=g)]
        for _ in range(repeat - 1):
            layers.append(ShuffleAddBlock(oup, g=g))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
