import os
import torch
import torch.nn as nn
from .core import blocks
from typing import Any

__all__ = ['ReXNet', 'rexnet_x0_9', 'rexnet_x1_0',
           'rexnet_x1_3', 'rexnet_x1_5', 'rexnet_x2_0',
           'rexnet_plain']

model_urls = {
    'rexnet_x0_9': None,
    'rexnet_x1_0': None,
    'rexnet_x1_3': None,
    'rexnet_x1_5': None,
    'rexnet_x2_0': None,
    'rexnet_plain': None
}


def _rexnet(
    arch: str,
    width_multiplier: float = 1.0,
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    model = ReXNet(width_multiplier=width_multiplier, **kwargs)

    if pretrained:
        if pth is not None:
            state_dict = torch.load(os.path.expanduser(pth))
        else:
            state_dict = torch.hub.load_state_dict_from_url(
                model_urls[arch],
                progress=progress
            )
        model.load_state_dict(state_dict)
    return model


def rexnet_x0_9(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _rexnet('rexnet_x0_9', 0.9, pretrained, pth, progress, **kwargs)


def rexnet_x1_0(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _rexnet('rexnet_x1_0', 1.0, pretrained, pth, progress, **kwargs)


def rexnet_x1_3(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _rexnet('rexnet_x1_3', 1.3, pretrained, pth, progress, **kwargs)


def rexnet_x1_5(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _rexnet('rexnet_x1_5', 1.5, pretrained, pth, progress, **kwargs)


def rexnet_x2_0(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _rexnet('rexnet_x2_0', 2.0, pretrained, pth, progress, **kwargs)


class InvertedResidualBlock(blocks.InvertedResidualBlock):
    def __init__(
        self,
        inp,
        oup,
        t, kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        se_ratio: float = None,
        se_ind: bool = True,
        dw_se_act: nn.Module = nn.ReLU6
    ):
        super().__init__(inp, oup, t, kernel_size=kernel_size, stride=stride,
                         padding=padding, se_ratio=se_ratio, se_ind=se_ind, dw_se_act=dw_se_act)

        self.apply_residual = (stride == 1) and (inp <= oup)
        self.branch2 = nn.Identity() if self.apply_residual else None
        self.combine = blocks.Combine('ADD') if self.apply_residual else None

    def forward(self, x):
        out = self.branch1(x)
        if self.apply_residual:
            out[:, 0:self.inp] += self.branch2(x)
        return out


class ReXNet(nn.Module):

    @blocks.nonlinear(nn.SiLU)
    @blocks.se(divisor=1, use_norm=True)
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        width_multiplier: float = 1.0,
        thumbnail: bool = False
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 2

        n = [2, 2, 3, 3, 5]  # repeats
        s = [FRONT_S, 2, 2, 1, 2]
        se = [0, 1/12, 1/12, 1/12, 1/12]

        self.depth = (sum(n[:]) + 1) * 3
        increase = 180 / (self.depth // 3 * 1.0)

        def multiplier(x): return int(round(x * width_multiplier))

        features = [
            blocks.Conv2dBlock(in_channels, multiplier(32), 3, FRONT_S),
            InvertedResidualBlock(multiplier(32), multiplier(16), 1)
        ]

        inplanes, planes = 16, 16 + increase
        for i, layers in enumerate(n):
            features.append(
                InvertedResidualBlock(
                    multiplier(inplanes), multiplier(planes), 6,
                    stride=s[i], se_ratio=se[i])
            )
            inplanes, planes = planes, planes + increase
            for _ in range(layers - 1):
                features.append(
                    InvertedResidualBlock(
                        multiplier(inplanes), multiplier(planes), 6, se_ratio=se[i])
                )
                inplanes, planes = planes, planes + increase

        features.append(blocks.Conv2d1x1Block(
            multiplier(inplanes), multiplier(1280)))

        self.features = nn.Sequential(*features)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(multiplier(1280), num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def rexnet_plain(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    model = ReXNetPlain(**kwargs)
    
    if pretrained:
        if pth is not None:
            state_dict = torch.load(os.path.expanduser(pth))
        else:
            state_dict = torch.hub.load_state_dict_from_url(
                model_urls['rexnet_plain'],
                progress=progress
            )
        model.load_state_dict(state_dict)
    return model


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


class ReXNetPlain(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        thumbnail: bool = False
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 2

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, 32, stride=FRONT_S,
                               activation_fn=nn.SiLU),
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

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
