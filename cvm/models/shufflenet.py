import torch
import torch.nn as nn
from .core import blocks, export, load_from_local_or_url
from typing import Any, OrderedDict, List


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


@export
class ShuffleNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        repeats: List[int] = [4, 84, 4],
        channels: List[int] = [],
        g: int = 3,
        thumbnail: bool = False,
        **kwargs: Any
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 2

        self.features = nn.Sequential(OrderedDict([
            ('stem', blocks.Conv2dBlock(in_channels, channels[0], 3, FRONT_S)),
            ('stage1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if not thumbnail else nn.Identity()),
            ('stage2', self.make_layers(repeats[0], channels[0], channels[1], g)),
            ('stage3', self.make_layers(repeats[1], channels[1], channels[2], g)),
            ('stage4', self.make_layers(repeats[2], channels[2], channels[3], g))
        ]))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels[3], num_classes)

    @staticmethod
    def make_layers(repeat, inp, oup, g):
        layers = [ShuffleCatBlock(inp, oup - inp, stride=2, g=g)]
        for _ in range(repeat - 1):
            layers.append(ShuffleAddBlock(oup, g=g))

        return blocks.Stage(layers)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _shufflenet(
    repeats: List[int],
    channels: List[int],
    g: int,
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    model = ShuffleNet(repeats=repeats, channels=channels, g=g, **kwargs)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
def shufflenet_g1(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _shufflenet([4, 8, 4], [24, 144, 288, 576], 1, pretrained, pth, progress, **kwargs)


@export
def shufflenet_g2(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _shufflenet([4, 8, 4], [24, 200, 400, 800], 2, pretrained, pth, progress, **kwargs)


@export
def shufflenet_g3(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _shufflenet([4, 8, 4], [24, 240, 480, 960], 3, pretrained, pth, progress, **kwargs)


@export
def shufflenet_g4(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _shufflenet([4, 8, 4], [24, 272, 544, 1088], 4, pretrained, pth, progress, **kwargs)


@export
def shufflenet_g8(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _shufflenet([4, 8, 4], [24, 384, 768, 1536], 8, pretrained, pth, progress, **kwargs)
