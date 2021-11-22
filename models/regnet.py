'''
Papers:
    [RegNet] Designing Network Design Spaces

RegNet trends:
    1. The depth of best models is stable across regimes, with an optimal depth of ~20 blocks(60 layers);
    2. The best models use a bottleneck ratio of 1.0, which effectively removes the bottleneck;
    3. The width multiplier wm of good models is ~2.5.

Notice:
    1. The inverted bottleneck degrades the EDF slightly and depthwise conv performs even worse relative to b = 1 and g >= 1.
    2. SE is useful;
    3. Swish outperforms ReLU at low flops, but ReLU is better at high flops. 
        Interestingly, if g is restricted to be 1 (depthwiseconv), Swish performs much better than ReLU.
'''
import os
import math
import torch
import torch.nn as nn
from .core import blocks, export, make_divisible, config
from typing import Any


class BottleneckTransform(nn.Sequential):
    @blocks.se(divisor=1)
    def __init__(
        self,
        inp,
        oup,
        stride,
        group_width,
        bottleneck_multiplier,
        se_ratio
    ):
        super().__init__()

        wb = int(round(oup * bottleneck_multiplier))

        self.add_module('1x1-1', blocks.Conv2d1x1Block(inp, wb))
        self.add_module('3x3', blocks.Conv2dBlock(wb, wb, stride=stride, groups=(wb // group_width)))

        if se_ratio:
            self.add_module('se', blocks.SEBlock(wb, (inp * se_ratio) / wb))  # se <-> inp

        self.add_module('1x1-2', blocks.Conv2d1x1BN(wb, oup))


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        group_width: int = 1,
        bottleneck_multiplier: float = 1.0,
        se_ratio: float = None,
    ) -> None:
        super().__init__()

        # Use skip connection with projection if shape changes
        self.proj = None
        should_proj = (inp != oup) or (stride != 1)
        if should_proj:
            self.proj = blocks.Conv2d1x1BN(inp, oup, stride)

        self.f = BottleneckTransform(
            inp,
            oup,
            stride,
            group_width,
            bottleneck_multiplier,
            se_ratio,
        )

        self.activation = blocks.activation_fn()

    def forward(self, x):
        if self.proj is not None:
            x = self.proj(x) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)


class RegStage(nn.Sequential):
    def __init__(
        self,
        in_width,
        out_width,
        stride,
        depth,
        group_widths,
        bottleneck_multiplier,
        se_ratio: float,
        stage_index: int
    ):
        super().__init__()

        for i in range(depth):
            self.add_module(
                f'block{stage_index}-{i}',
                ResBottleneckBlock(
                    in_width if i == 0 else out_width,
                    out_width,
                    stride if i == 0 else 1,
                    group_widths,
                    bottleneck_multiplier,
                    se_ratio
                )
            )


@export
class RegNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        stem_width: int = 32,
        d: int = None,
        w0: int = None,
        wa: float = None,
        wm: float = None,
        b: float = None,
        g: int = None,
        se_ratio: float = None,
        dropout_rate: float = 0.0,
        **kwargs: Any
    ):
        """
        d: the number of blocks
        w0: initial width
        wa: slope
            uj = w0 + wa * j  for 0 <= j < d -> for each block
        wm: 
        b: bottleneck ratio
        g: group width
        """
        super().__init__()

        self.features = nn.Sequential()
        self.features.add_module('stem', blocks.Conv2dBlock(in_channels, stem_width, stride=2))

        uj = w0 + wa * torch.arange(d)
        sj = torch.round(torch.log(uj / w0) / math.log(wm))

        widths = (torch.round((w0 * torch.pow(wm, sj)) / 8) * 8).int().tolist()
        widths = [int(make_divisible(w * b, min(g, w * b)) / b) for w in widths] # Adjusts the compatibility of widths and groups
        num_stages = len(set(widths))
        stage_depths = [(torch.tensor(widths) == w).sum().item() for w in torch.unique(torch.tensor(widths))]
        stage_widths = torch.unique(torch.tensor(widths)).numpy().tolist()
        group_widths = [g] * num_stages
        group_widths = [min(g, int(w * b)) for g, w in zip(group_widths, stage_widths)]
        bottleneck_multipliers = [b] * num_stages

        stage_widths = [stem_width] + stage_widths

        for i in range(num_stages):
            self.features.add_module(
                f'stage{i}',
                RegStage(
                    stage_widths[i],
                    stage_widths[i+1],
                    2,
                    stage_depths[i],
                    group_widths[i],
                    bottleneck_multipliers[i],
                    se_ratio,
                    i + 1
                )
            )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate, inplace=True),
            nn.Linear(stage_widths[-1], num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _regnet(
    d: int,
    w0: int,
    wa: float,
    wm: float,
    b: float = 1.0,
    g: int = None,
    se_ratio: float = None,
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    model = RegNet(d=d, w0=w0, wa=wa, wm=wm, b=b, g=g, se_ratio=se_ratio, **kwargs)

    if pretrained:
        if pth is not None:
            state_dict = torch.load(os.path.expanduser(pth))
        else:
            assert 'url' in kwargs and kwargs['url'] != '', 'Invalid URL.'
            state_dict = torch.hub.load_state_dict_from_url(
                kwargs['url'],
                progress=progress
            )
        model.load_state_dict(state_dict)
    return model


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.1-regnets/regnet_x_400mf-903d111f.pth')
def regnet_x_400mf(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs):
    return _regnet(22, 24, 24.48, 2.54, 1.0, 16, None, pretrained, pth, progress, **kwargs)


@export
def regnet_x_800mf(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs):
    return _regnet(16, 56, 35.73, 2.28, 1.0, 16, None, pretrained, pth, progress, **kwargs)


@export
def regnet_x_1_6gf(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs):
    return _regnet(18, 80, 34.01, 2.25, 1.0, 24, None, pretrained, pth, progress, **kwargs)


@export
def regnet_x_3_2gf(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs):
    return _regnet(25, 88, 26.32, 2.25, 1.0, 48, None, pretrained, pth, progress, **kwargs)


@export
def regnet_x_8gf(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs):
    return _regnet(23, 80, 49.56, 2.88, 1.0, 120, None, pretrained, pth, progress, **kwargs)


@export
def regnet_x_16gf(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs):
    return _regnet(22, 216, 55.59, 2.1, 1.0, 128, None, pretrained, pth, progress, **kwargs)


@export
def regnet_x_32gf(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs):
    return _regnet(23, 320, 69.86, 2.0, 1.0, 168, None, pretrained, pth, progress, **kwargs)


@export
def regnet_y_400mf(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs):
    return _regnet(16, 48, 27.89, 2.09, 1.0, 8, 0.25, pretrained, pth, progress, **kwargs)


@export
def regnet_y_800mf(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs):
    return _regnet(14, 56, 38.84, 2.4, 1.0, 16, 0.25, pretrained, pth, progress, **kwargs)


@export
def regnet_y_1_6gf(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs):
    return _regnet(27, 48, 20.71, 2.65, 1.0, 24, 0.25, pretrained, pth, progress, **kwargs)


@export
def regnet_y_3_2gf(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs):
    return _regnet(21, 80, 42.63, 2.66, 1.0, 24, 0.25, pretrained, pth, progress, **kwargs)


@export
def regnet_y_8gf(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs):
    return _regnet(17, 192, 76.82, 2.19, 1.0, 56, 0.25, pretrained, pth, progress, **kwargs)


@export
def regnet_y_16gf(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs):
    return _regnet(18, 200, 106.23, 2.48, 1.0, 112, 0.25, pretrained, pth, progress, **kwargs)


@export
def regnet_y_32gf(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs):
    return _regnet(20, 232, 115.89, 2.53, 1.0, 232, 0.25, pretrained, pth, progress, **kwargs)
