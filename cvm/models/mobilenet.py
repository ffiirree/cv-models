import torch
import torch.nn as nn
from .core import blocks, export, config, load_from_local_or_url
from typing import Any, OrderedDict, Type, Union, List


class MobileBlock(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        groups: int = 1,
        ratio: float = 0.75  # unused
    ):
        super().__init__(
            blocks.DepthwiseBlock(inp, inp, kernel_size, stride, padding, dilation=dilation),
            blocks.PointwiseBlock(inp, oup, groups=groups)
        )


class DepthSepBlock(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        groups: int = 1,
        ratio: float = 0.75  # unused
    ):
        super().__init__(
            blocks.DepthwiseConv2d(inp, inp, kernel_size, stride, padding, dilation=dilation),
            blocks.PointwiseBlock(inp, oup, groups=groups)
        )


class SDBlock(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        groups: int = 1,
        ratio: float = 0.75
    ):
        if ratio is None or stride > 1:
            super().__init__(
                blocks.GaussianBlurBlock(inp, kernel_size, stride, padding=padding, dilation=dilation),
                blocks.PointwiseBlock(inp, oup, groups=groups)
            )
        else:
            super().__init__(
                blocks.SDDCBlock(inp, stride, dilation=dilation, ratio=ratio),
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
        dilations: List[int] = None,
        thumbnail: bool = False,
        **kwargs: Any
    ):
        super().__init__()

        def depth(d): return max(int(d * depth_multiplier), 8)

        dilations = dilations or [1, 1, 1, 1]
        assert len(dilations) == 4, ''

        FRONT_S = 1 if thumbnail else 2

        layers = [2, 2, 6, 2]
        strides = [FRONT_S, 2, 2, 2]
        ratios = [7/8, 6/8, 4/8, 4/8, 1/16]

        self.features = nn.Sequential(OrderedDict([
            ('stem', blocks.Stage(
                blocks.Conv2dBlock(in_channels, depth(base_width), stride=FRONT_S),
                block(depth(base_width), depth(base_width) * 2, ratio=ratios[0])
            ))
        ]))

        for stage, stride in enumerate(strides):
            inp = depth(base_width * 2 ** (stage + 1))
            oup = depth(base_width * 2 ** (stage + 2))

            self.features.add_module(f'stage{stage+1}', blocks.Stage(
                [block(
                    inp if i == 0 else oup,
                    oup,
                    stride=stride if (i == 0 and dilations[stage] == 1) else 1,
                    dilation=max(dilations[stage] // (stride if i == 0 else 1), 1),
                    ratio=None if stride != 1 and i == 0 else ratios[stage+1]
                ) for i in range(layers[stage])]
            ))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate, inplace=True),
            nn.Linear(oup, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
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
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.1/mobilenet_v1_x1_0-e00006ef.pth')
def mobilenet_v1_x1_0(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mobilenet_v1(1.0, MobileBlock, pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.1/mobilenet_v1_x0_75-43c1cb04.pth')
def mobilenet_v1_x0_75(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mobilenet_v1(0.75, MobileBlock, pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.1/mobilenet_v1_x0_5-588ee141.pth')
def mobilenet_v1_x0_5(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mobilenet_v1(0.5, MobileBlock, pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.1/mobilenet_v1_x0_35-cbab38a6.pth')
def mobilenet_v1_x0_35(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mobilenet_v1(0.35, MobileBlock, pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.1/mobilenet_v1_x1_0_wo_dwrelubn-2956d795.pth')
@blocks.normalizer(position='after')
def mobilenet_v1_x1_0_wo_dwrelubn(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs):
    return _mobilenet_v1(1.0, DepthSepBlock, pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.1/sd_mobilenet_v1_x1_0-c5bd0c22.pth')
def sd_mobilenet_v1_x1_0(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mobilenet_v1(1.0, SDBlock, pretrained, pth, progress, **kwargs)
