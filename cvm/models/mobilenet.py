import os
import torch
import torch.nn as nn
from .core import blocks, export, config
from typing import Any, Type, Union


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


@export
class MobileNet(nn.Module):
    '''https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py'''

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        base_width: int = 32,
        block: Type[Union[MobileBlock, DepthSepBlock]] = MobileBlock,
        depth_multiplier: float = 1.0,
        dropout_rate: float = 0.2,
        thumbnail: bool = False,
        **kwargs: Any
    ):
        super().__init__()

        def depth(d): return max(int(d * depth_multiplier), 8)

        FRONT_S = 1 if thumbnail else 2

        strides = [1, FRONT_S, 1, 2, 1, 2,  1,  1,  1,  1,  1,  2,  1]
        factors = [1, 2, 4, 4, 8, 8, 16, 16, 16, 16, 16, 16, 32, 32]

        layers = [blocks.Conv2dBlock(in_channels, depth(base_width), stride=FRONT_S)]

        for i, s in enumerate(strides):
            inp = depth(base_width * factors[i])
            oup = depth(base_width * factors[i + 1])
            layers.append(block(inp, oup, stride=s))

        self.features = nn.Sequential(*layers)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate, inplace=True),
            nn.Linear(depth(base_width * factors[-1]), num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _mobilenet_v1(
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
            assert 'url' in kwargs and kwargs['url'] != '', 'Invalid URL.'
            state_dict = torch.hub.load_state_dict_from_url(
                kwargs['url'],
                progress=progress
            )
        model.load_state_dict(state_dict)
    return model


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.1/mobilenet_v1_x1_0-b6e1e34f.pth')
def mobilenet_v1_x1_0(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mobilenet_v1(1.0, MobileBlock, pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.1/mobilenet_v1_x0_75-82d76756.pth')
def mobilenet_v1_x0_75(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mobilenet_v1(0.75, MobileBlock, pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.1/mobilenet_v1_x0_5-0fbeb3fb.pth')
def mobilenet_v1_x0_5(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mobilenet_v1(0.5, MobileBlock, pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.1/mobilenet_v1_x0_35-16b60798.pth')
def mobilenet_v1_x0_35(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mobilenet_v1(0.35, MobileBlock, pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.1/mobilenet_v1_x1_0_wo_dwrelubn-9cd9f96e.pth')
@blocks.normalizer(position='after')
def mobilenet_v1_x1_0_wo_dwrelubn(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs):
    return _mobilenet_v1(1.0, DepthSepBlock, pretrained, pth, progress, **kwargs)
