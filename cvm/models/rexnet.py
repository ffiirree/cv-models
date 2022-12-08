from functools import partial
import torch
import torch.nn as nn

from .ops import blocks
from .utils import export, config, load_from_local_or_url
from typing import Any


class InvertedResidualBlock(blocks.InvertedResidualBlock):
    def __init__(
        self,
        inp,
        oup,
        t, kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        rd_ratio: float = None,
        se_ind: bool = True,
        dw_se_act: nn.Module = nn.ReLU6
    ):
        super().__init__(inp, oup, t, kernel_size=kernel_size, stride=stride,
                         padding=padding, rd_ratio=rd_ratio, se_ind=se_ind, dw_se_act=dw_se_act)

        self.apply_residual = (stride == 1) and (inp <= oup)
        self.branch2 = nn.Identity() if self.apply_residual else None
        self.combine = blocks.Combine('ADD') if self.apply_residual else None

    def forward(self, x):
        out = self.branch1(x)
        if self.apply_residual:
            out[:, 0:self.inp] += self.branch2(x)
        return out


@export
class ReXNet(nn.Module):

    @blocks.activation(partial(nn.SiLU, inplace=True))
    @blocks.se(divisor=1, use_norm=True)
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        width_multiplier: float = 1.0,
        dropout_rate: float = 0.2,
        thumbnail: bool = False,
        **kwargs: Any
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 2

        n = [2, 2, 3, 3, 5]  # repeats
        s = [FRONT_S, 2, 2, 1, 2]
        ratios = [0, 1/12, 1/12, 1/12, 1/12]

        self.depth = (sum(n[:]) + 1) * 3
        increase = 180 / (self.depth // 3 * 1.0)

        def multiplier(x): return int(round(x * width_multiplier))

        features = [
            blocks.Conv2dBlock(in_channels, multiplier(32), 3, FRONT_S),
            InvertedResidualBlock(multiplier(32), multiplier(16), 1)
        ]

        inplanes, planes = 16, 16 + increase
        for i, layers in enumerate(n):
            features.append(InvertedResidualBlock(multiplier(inplanes),
                            multiplier(planes), 6, stride=s[i], rd_ratio=ratios[i]))
            inplanes, planes = planes, planes + increase
            for _ in range(layers - 1):
                features.append(InvertedResidualBlock(multiplier(inplanes), multiplier(planes), 6, rd_ratio=ratios[i]))
                inplanes, planes = planes, planes + increase

        features.append(blocks.Conv2d1x1Block(multiplier(inplanes), multiplier(1280)))

        self.features = nn.Sequential(*features)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(multiplier(1280), num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _rexnet(
    width_multiplier: float = 1.0,
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    model = ReXNet(width_multiplier=width_multiplier, **kwargs)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
def rexnet_x0_9(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _rexnet(0.9, pretrained, pth, progress, **kwargs)


@export
def rexnet_x1_0(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _rexnet(1.0, pretrained, pth, progress, **kwargs)


@export
def rexnet_x1_3(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _rexnet(1.3, pretrained, pth, progress, **kwargs)


@export
def rexnet_x1_5(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _rexnet(1.5, pretrained, pth, progress, **kwargs)


@export
def rexnet_x2_0(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _rexnet(2.0, pretrained, pth, progress, **kwargs)


class PlainBlock(nn.Sequential):
    def __init__(self, inplanes, planes, stride: int = 1):
        super().__init__(
            blocks.DepthwiseConv2d(inplanes, inplanes, stride=stride),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            blocks.PointwiseBlock(inplanes, planes),
            nn.BatchNorm2d(planes),
            nn.SiLU(inplace=True)
        )


@export
class ReXNetPlain(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        dropout_rate: float = 0.2,
        thumbnail: bool = False,
        **kwargs: Any
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 2

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, 32, stride=FRONT_S, activation_fn=partial(nn.SiLU, inplace=True)),
            PlainBlock(32, 96, stride=FRONT_S),
            PlainBlock(96, 144),
            PlainBlock(144, 192, stride=2),
            PlainBlock(192, 240),
            PlainBlock(240, 288, stride=2),
            PlainBlock(288, 336),
            PlainBlock(336, 384),
            PlainBlock(384, 432),
            PlainBlock(432, 480),
            PlainBlock(480, 528),
            PlainBlock(528, 576, stride=2),
            PlainBlock(576, 624),
            PlainBlock(624, 1024),
            blocks.Conv2d1x1Block(1024, 1280)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


@export
def rexnet_plain(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    model = ReXNetPlain(**kwargs)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model
