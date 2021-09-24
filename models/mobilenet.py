import os
import torch
import torch.nn as nn
from .core import blocks
from typing import Any, Type, Union

__all__ = ['MobileNet',
           'mobilenet_lineardw', 'mobilenet_v1_x1_0', 'mobilenet_v1_x0_75',
           'mobilenet_v1_x0_5', 'mobilenet_v1_x0_35']


model_urls = {
    'mobilenet_v1_x1_0': 'https://github.com/ffiirree/models/releases/download/v0.0.1/mobilenet_v1_x1_0-72d57278.pth',
    'mobilenet_v1_x0_75': 'https://github.com/ffiirree/models/releases/download/v0.0.1/mobilenet_v1_x0_75-82d76756.pth',
    'mobilenet_v1_x0_5': 'https://github.com/ffiirree/models/releases/download/v0.0.1/mobilenet_v1_x0_5-824260cb.pth',
    'mobilenet_v1_x0_35': None,
    'mobilenet_lineardw': None
}


class MobileBlock(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1
    ):
        super().__init__(
            blocks.DepthwiseBlock(inp, inp, kernel_size, stride, padding),
            blocks.PointwiseBlock(inp, oup, groups=groups)
        )


class DepthSepBlock(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1
    ):
        super().__init__(
            blocks.DepthwiseConv2d(inp, inp, kernel_size, stride, padding),
            blocks.PointwiseBlock(inp, oup, groups=groups)
        )


def _mobilenet_v1(
    arch: str,
    depth_multiplier: float = 1.0,
    block: Type[Union[MobileBlock, DepthSepBlock]] = MobileBlock,
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    model = MobileNet(depth_multiplier=depth_multiplier, block=block, **kwargs)

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


def mobilenet_v1_x1_0(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mobilenet_v1('mobilenet_v1_x1_0', 1.0, MobileBlock, pretrained, pth, progress, **kwargs)


def mobilenet_v1_x0_75(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mobilenet_v1('mobilenet_v1_x0_75', 1.0, MobileBlock, pretrained, pth, progress, **kwargs)


def mobilenet_v1_x0_5(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mobilenet_v1('mobilenet_v1_x0_5', 1.0, MobileBlock, pretrained, pth, progress, **kwargs)


def mobilenet_v1_x0_35(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mobilenet_v1('mobilenet_v1_x0_35', 1.0, MobileBlock, pretrained, pth, progress, **kwargs)


def mobilenet_lineardw(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs):
    return _mobilenet_v1('mobilenet_v1_x1_0', 1.0, DepthSepBlock, pretrained, pth, progress, **kwargs)


class MobileNet(nn.Module):
    '''https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py'''

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        base_width: int = 32,
        block: Type[Union[MobileBlock, DepthSepBlock]] = MobileBlock,
        depth_multiplier: float = 1.0,
        thumbnail: bool = False
    ):
        super().__init__()

        def depth(d): return max(int(d * depth_multiplier), 8)

        FRONT_S = 1 if thumbnail else 2

        strides = [1, 2, 1, 2, 1, 2,  1,  1,  1,  1,  1,  2,  1]
        factors = [1, FRONT_S, 4, 4, 8, 8, 16, 16, 16, 16, 16, 16, 32, 32]

        layers = [
            blocks.Conv2dBlock(in_channels, depth(base_width), stride=FRONT_S)
        ]

        for i, s in enumerate(strides):
            inp = depth(base_width * factors[i])
            oup = depth(base_width * factors[i + 1])
            layers.append(block(inp, oup, stride=s))

        self.features = nn.Sequential(*layers)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(depth(base_width * 32), num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
