from functools import partial
import torch
import torch.nn as nn
from .core import blocks, make_divisible, export, config, load_from_local_or_url
from typing import Any, OrderedDict, Type, Union, List


@export
class MobileNetV2(nn.Module):
    @blocks.nonlinear(partial(nn.ReLU6, inplace=True))
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        multiplier: float = 1.0,
        block: Type[Union[blocks.InvertedResidualBlock, blocks.SDInvertedResidualBlock]] = blocks.InvertedResidualBlock,
        dropout_rate: float = 0.2,
        dilations: List[int] = None,
        thumbnail: bool = False,
        **kwargs: Any
    ):
        super().__init__()

        dilations = [1] + (dilations or [1, 1, 1, 1])
        assert len(dilations) == 5, ''

        self.block = block

        FRONT_S = 1 if thumbnail else 2

        t = [1, 6, 6, 6, 6, 6, 6]
        c = [32, 16, 24, 32, 64, 96, 160, 320]
        n = [1, 2, 3, 4, 3, 3, 1]
        s = [1, FRONT_S, 2, 2, 1, 2, 1]
        stages = [0, 1, 1, 1, 0, 1, 0]
        ratios = [7/8, 7/8, 6/8, 5/8, 4/8, 4/8, 1/16]

        if multiplier < 1.0:
            c = [make_divisible(x * multiplier, 8) for x in c]

        self.features = nn.Sequential(OrderedDict([
            ('stem', blocks.Stage(
                blocks.Conv2dBlock(in_channels, c[0], 3, stride=FRONT_S)
            ))
        ]))

        for i in range(len(t)):
            layers = self.make_layers(
                c[i],
                t[i],
                c[i+1],
                n[i],
                s[i],
                dilations[len(self.features) + (stages[i] - 1)],
                ratios[i]
            )

            if stages[i]:
                self.features.add_module(f'stage{len(self.features)}', blocks.Stage(layers))
            else:
                self.features[-1].append(layers)

        self.features[-1].append(blocks.Conv2d1x1Block(c[-1], 1280))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate, inplace=True),
            nn.Linear(1280, num_classes)
        )

    def make_layers(self, inp: int, t: int, oup: int, n: int, stride: int, dilation: int, ratio: float):
        layers = [self.block(inp, oup, t, stride=stride if dilation == 1 else 1,
                             dilation=max(dilation // stride, 1), ratio=ratio if stride == 1 else None)]

        for _ in range(n - 1):
            layers.append(self.block(oup, oup, t, dilation=dilation, ratio=ratio))

        return layers

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def _mobilenet_v2(
    multiplier: float = 1.0,
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    model = MobileNetV2(multiplier=multiplier, **kwargs)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.1/mobilenet_v2_x1_0-bf342af4.pth')
def mobilenet_v2_x1_0(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mobilenet_v2(1.0, pretrained, pth, progress, **kwargs)


@export
def sd_mobilenet_v2_x1_0(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['block'] = blocks.SDInvertedResidualBlock
    return _mobilenet_v2(1.0, pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.1/mobilenet_v2_x0_75-fdfaf351.pth')
def mobilenet_v2_x0_75(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mobilenet_v2(0.75, pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.1/mobilenet_v2_x0_5-a9d4ed71.pth')
def mobilenet_v2_x0_5(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mobilenet_v2(0.5, pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.1/mobilenet_v2_x0_35-9bce1f31.pth')
def mobilenet_v2_x0_35(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mobilenet_v2(0.35, pretrained, pth, progress, **kwargs)
