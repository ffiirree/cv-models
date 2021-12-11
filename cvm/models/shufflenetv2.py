import torch
import torch.nn as nn
from .core import blocks, export, load_from_local_or_url
from typing import Any, OrderedDict, List


class ShuffleBlockV2(nn.Module):
    def __init__(self, inp, oup, stride: int = 1):
        super().__init__()

        self.inp = inp
        self.oup = oup // 2
        self.stride = stride
        self.split = None

        if self.stride == 1:
            self.inp = inp // 2
            self.split = blocks.ChannelChunk(2)

        self.branch1 = nn.Identity()
        if self.stride != 1:
            self.branch1 = nn.Sequential(OrderedDict([
                ('dwconv', blocks.DepthwiseConv2dBN(self.inp, self.inp, stride=self.stride)),
                ('1x1', blocks.Conv2d1x1Block(self.inp, self.oup))
            ]))

        self.branch2 = nn.Sequential(OrderedDict([
            ('1x1-1', blocks.Conv2d1x1Block(self.inp, self.oup)),
            ('dwconv', blocks.DepthwiseConv2dBN(self.oup, self.oup, stride=self.stride)),
            ('1x1-2', blocks.Conv2d1x1Block(self.oup, self.oup))
        ]))

        self.combine = blocks.Combine('CONCAT')
        self.shuffle = blocks.ChannelShuffle(groups=2)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = self.split(x)
            x2 = self.branch2(x2)
        else:
            x1 = self.branch1(x)
            x2 = self.branch2(x)

        out = self.combine([x1, x2])
        out = self.shuffle(out)
        return out


@export
class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        repeats: List[int] = [4, 8, 4],
        channels: List[int] = [24, 48, 96, 192, 1024],
        thumbnail: bool = False,
        **kwargs: Any
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 2

        self.features = nn.Sequential(OrderedDict([
            ('stem', blocks.Conv2dBlock(in_channels, channels[0], 3, FRONT_S)),
            ('stage1', nn.MaxPool2d(3, stride=2, padding=1) if not thumbnail else nn.Identity()),
            ('stage2', self.make_layers(repeats[0], channels[0], channels[1])),
            ('stage3', self.make_layers(repeats[1], channels[1], channels[2])),
            ('stage4', self.make_layers(repeats[2], channels[2], channels[3])),
        ]))
        
        self.features.stage4.append(
            blocks.Conv2d1x1Block(channels[3], channels[4])
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels[4], num_classes)

    @staticmethod
    def make_layers(repeat, inp, oup):
        layers = [ShuffleBlockV2(inp, oup, stride=2)]
        for _ in range(repeat - 1):
            layers.append(ShuffleBlockV2(oup, oup))

        return blocks.Stage(layers)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _shufflenet_v2(
    repeats: List[int],
    channels: List[int],
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    model = ShuffleNetV2(repeats=repeats, channels=channels, **kwargs)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
def shufflenet_v2_x0_5(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _shufflenet_v2([4, 8, 4], [24, 48, 96, 192, 1024], pretrained, pth, progress, **kwargs)


@export
def shufflenet_v2_x1_0(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _shufflenet_v2([4, 8, 4], [24, 116, 232, 464, 1024], pretrained, pth, progress, **kwargs)


@export
def shufflenet_v2_x1_5(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _shufflenet_v2([4, 8, 4], [24, 176, 352, 704, 1024], pretrained, pth, progress, **kwargs)


@export
def shufflenet_v2_x2_0(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _shufflenet_v2([4, 8, 4], [24, 244, 488, 976, 2048], pretrained, pth, progress, **kwargs)
