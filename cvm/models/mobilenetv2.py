from functools import partial
import torch
import torch.nn as nn
from .core import blocks, make_divisible, export, config, load_from_local_or_url
from typing import Any


@export
class MobileNetV2(nn.Module):
    @blocks.nonlinear(partial(nn.ReLU6, inplace=True))
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        multiplier: float = 1.0,
        dropout_rate: float = 0.2,
        thumbnail: bool = False,
        **kwargs: Any
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 2

        t = [1, 6, 6, 6, 6, 6, 6]
        c = [32, 16, 24, 32, 64, 96, 160, 320]
        n = [1, 2, 3, 4, 3, 3, 1]
        s = [1, FRONT_S, 2, 2, 1, 2, 1]

        if multiplier < 1.0:
            c = [make_divisible(x * multiplier, 8) for x in c]

        features = [blocks.Conv2dBlock(in_channels, c[0], 3, stride=FRONT_S)]

        for i in range(len(t)):
            features.append(self.make_layers(c[i], t[i], c[i+1], n[i], s[i]))

        features.append(blocks.Conv2d1x1Block(c[-1], 1280))

        self.features = nn.Sequential(*features)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate, inplace=True),
            nn.Linear(1280, num_classes)
        )

    @staticmethod
    def make_layers(inp: int, t: int, oup: int, n: int, stride: int):
        layers = [blocks.InvertedResidualBlock(inp, oup, t, stride=stride)]

        for _ in range(n - 1):
            layers.append(blocks.InvertedResidualBlock(oup, oup, t))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
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
def mobilenet_v2_x1_0(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mobilenet_v2(1.0, pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.1/mobilenet_v2_x0_75-144da943.pth')
def mobilenet_v2_x0_75(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mobilenet_v2(0.75, pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.1/mobilenet_v2_x0_5-1e1467ed.pth')
def mobilenet_v2_x0_5(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mobilenet_v2(0.5, pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.1/mobilenet_v2_x0_35-cc1f8697.pth')
def mobilenet_v2_x0_35(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mobilenet_v2(0.35, pretrained, pth, progress, **kwargs)
