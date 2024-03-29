"""https://github.com/huawei-noah/CV-Backbones/blob/master/ghostnet_pytorch/ghostnet.py"""
import math
import torch
import torch.nn as nn

from .ops import blocks
from .utils import export, config, load_from_local_or_url
from .ops.functional import make_divisible
from typing import Any, List


class GhostModule(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        kernel_size=1,
        ratio=2,
        dw_size=3,
        stride=1,
        relu=True
    ):
        super().__init__()
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        dw_kernel_size: int = 3,
        stride: int = 1,
        act_layer: nn.Module = nn.ReLU,
        rd_ratio: float = 0.
    ):
        super(GhostBottleneck, self).__init__()
        has_attn = rd_ratio is not None and rd_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs, mid_chs, dw_kernel_size, stride=stride,
                padding=(dw_kernel_size-1)//2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_attn:
            self.se = blocks.SEBlock(mid_chs, rd_ratio=rd_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


@export
class GhostNet(nn.Module):
    @blocks.se(gating_fn=nn.Hardsigmoid)
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        multiplier: float = 1.0,
        dropout_rate: float = 0.2,
        cfgs: List[list] = [],
        thumbnail: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__()

        FRONT_S = 1 if thumbnail else 2

        inp = make_divisible(16 * multiplier, 4)
        features = [blocks.Conv2dBlock(in_channels, inp, stride=FRONT_S)]

        for cfg in cfgs:
            stage = blocks.Stage()
            for k, t, c, rd_ratio, s in cfg:
                oup = make_divisible(c * multiplier, 4)
                stage.append(GhostBottleneck(
                    inp, make_divisible(t * multiplier, 4), oup, k, s, rd_ratio=rd_ratio
                ))
                inp = oup

            features.append(stage)

        oup = make_divisible(t * multiplier, 4)
        features.append(blocks.Conv2d1x1Block(inp, oup))

        self.features = nn.Sequential(*features)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(oup, 1280),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate, inplace=True),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _ghostnet(
    multiplier: float = 1.0,
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3,  16,  16, 0, 1]],
        # stage2
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],
        # stage3
        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],
        # stage4
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
         ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
         ]
    ]

    model = GhostNet(multiplier=multiplier, cfgs=cfgs, **kwargs)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
def ghostnet_x0_5(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs):
    return _ghostnet(0.5, pretrained, pth, progress, **kwargs)


@export
def ghostnet_x1_0(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs):
    return _ghostnet(1.0, pretrained, pth, progress, **kwargs)


@export
def ghostnet_x1_3(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs):
    return _ghostnet(1.3, pretrained, pth, progress, **kwargs)
