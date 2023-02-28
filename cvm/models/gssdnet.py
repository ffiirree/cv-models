import torch
import torch.nn as nn

from .ops import blocks
from .utils import export, config, load_from_local_or_url
from typing import Any, OrderedDict, Tuple
import torch.nn.functional as F


class GaussianBlurBlock(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        sigma_range: Tuple[float, float] = (1.0, 1.0),
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        groups: int = 1
    ) -> None:
        super().__init__(
            blocks.GaussianBlur(inp, kernel_size, sigma_range, stride=stride, padding=padding, dilation=dilation),
            blocks.normalizer_fn(inp),
            blocks.PointwiseBlock(inp, oup, groups=groups)
        )


class PartialGaussianBlur(blocks.GaussianBlur):
    def __init__(
        self,
        channels,
        kernel_size: int = 3,
        sigma_range: Tuple[float, float] = (0.707, 1.414),
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        ratio: float = 1.0
    ) -> None:

        super().__init__(int(channels * ratio), kernel_size, sigma_range, True, stride, padding, dilation)

    def forward(self, x):
        return torch.cat([super().forward(x[:, :self.channels]), x[:, self.channels:]], dim=1)


class LowPassFilterBlock(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        groups: int = 1,
        ratio: float = 0.5
    ):
        super().__init__(
            PartialGaussianBlur(inp, kernel_size, stride=stride, padding=padding, dilation=dilation, ratio=ratio),
            blocks.normalizer_fn(inp),
            blocks.PointwiseBlock(inp, oup, groups=groups)
        )


class Order2Conv2d(nn.Module):
    def __init__(
        self,
        inp,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        ratio: float = 1.0
    ) -> None:
        super().__init__()

        if padding is None:
            padding = ((kernel_size - 1) * (dilation - 1) + kernel_size) // 2

        self.in_channels = inp
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ratio = ratio

        self.inp = (((int)(self.in_channels * ratio)) // 5) * 5

        kernels = torch.tensor([
            [[
                [0, 0,  0],
                [1, 0, -1],
                [0, 0,  0]
            ]],
            [[
                [0, -1, 0],
                [0,  0, 0],
                [0,  1, 0]
            ]],
            [[
                [0, 0,  0],
                [-1, 2, -1],
                [0, 0,  0]
            ]],
            [[
                [0, -1, 0],
                [0,  2, 0],
                [0, -1, 0]
            ]],
            [[
                [-1, 0, 1],
                [0, 0,  0],
                [1, 0,  -1]
            ]],
        ], dtype=torch.float32).repeat(self.inp // 5, 1, 1, 1)

        self.weight = nn.Parameter(kernels, False)

    def forward(self, x):
        return torch.cat([
            F.conv2d(x[:, :self.inp], self.weight, None, self.stride, self.padding, self.dilation, self.inp),
            x[:, self.inp:]
        ], dim=1)

    def extra_repr(self):
        return f'channels={self.in_channels}, ratio={self.ratio}({self.inp})'


class Order1Conv2d(nn.Module):
    def __init__(
        self,
        inp,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        ratio: float = 1.0
    ) -> None:
        super().__init__()

        if padding is None:
            padding = ((kernel_size - 1) * (dilation - 1) + kernel_size) // 2

        self.in_channels = inp
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ratio = ratio

        self.inp = (((int)(self.in_channels * ratio)) // 2) * 2

        kernels = torch.tensor([
            [[
                [0, 0,  0],
                [1, 0, -1],
                [0, 0,  0]
            ]],
            [[
                [0, -1, 0],
                [0,  0, 0],
                [0,  1, 0]
            ]]
        ], dtype=torch.float32).repeat(self.inp // 2, 1, 1, 1)

        self.weight = nn.Parameter(kernels, False)

    def forward(self, x):
        return torch.cat([
            F.conv2d(x[:, :self.inp], self.weight, None, self.stride, self.padding, self.dilation, self.inp),
            x[:, self.inp:]
        ], dim=1)

    def extra_repr(self):
        return f'channels={self.in_channels}, ratio={self.ratio}({self.inp})'


class Order2Block(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        groups: int = 1,
        ratio: float = 0.5
    ):
        super().__init__(
            Order2Conv2d(inp, kernel_size, stride, padding, dilation=dilation, ratio=ratio),
            blocks.normalizer_fn(inp),
            blocks.PointwiseBlock(inp, oup, groups=groups)
        )


class Order1Block(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        groups: int = 1,
        ratio: float = 0.5
    ):
        super().__init__(
            Order1Conv2d(inp, kernel_size, stride, padding, dilation=dilation, ratio=ratio),
            blocks.normalizer_fn(inp),
            blocks.PointwiseBlock(inp, oup, groups=groups)
        )


@export
class GSSDNet(nn.Module):
    """
        Paper: [Frequency and Scale Perspectives of Feature Extraction](https://arxiv.org/abs/2302.12477)
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        base_width: int = 32,
        depth_multiplier: float = 1.0,
        derivate_fn: nn.Module = Order2Block,
        downsample_fn: nn.Module = GaussianBlurBlock,
        lowpass_fn: nn.Module = LowPassFilterBlock,
        dropout_rate: float = 0.2,
        thumbnail: bool = False,
        **kwargs: Any
    ):
        super().__init__()

        def depth(d): return max(int(d * depth_multiplier), 8)

        FRONT_S = 1 if thumbnail else 2

        strides = [FRONT_S, FRONT_S, 2, 2, 2]
        ratios = [4/8, 4/8, 4/8, 4/8, 4/8]

        self.features = nn.Sequential(OrderedDict([
            ('stem', blocks.Stage(
                downsample_fn(in_channels, depth(base_width), 3, stride=strides[0]),
                derivate_fn(depth(base_width), depth(base_width) * 2, ratio=ratios[0])
            )),
            ('stage1', blocks.Stage(
                downsample_fn(depth(base_width * 2), depth(base_width * 4), 3, stride=strides[1]),
                derivate_fn(depth(base_width * 4), depth(base_width * 4), ratio=ratios[1])
            )),
            ('stage2', blocks.Stage(
                downsample_fn(depth(base_width * 4), depth(base_width * 8), 3, stride=strides[2]),
                derivate_fn(depth(base_width * 8), depth(base_width * 8), ratio=ratios[2])
            )),
            ('stage3', blocks.Stage(
                downsample_fn(depth(base_width * 8), depth(base_width * 16), 3, stride=strides[3]),
                derivate_fn(depth(base_width * 16), depth(base_width * 16), ratio=ratios[3]),

                lowpass_fn(depth(base_width * 16), depth(base_width * 16), ratio=ratios[3]),
                derivate_fn(depth(base_width * 16), depth(base_width * 16), ratio=ratios[3]),

                lowpass_fn(depth(base_width * 16), depth(base_width * 16), ratio=ratios[3]),
                derivate_fn(depth(base_width * 16), depth(base_width * 16), ratio=ratios[3]),
            )),
            ('stage4', blocks.Stage(
                downsample_fn(depth(base_width * 16), depth(base_width * 32), 3, stride=strides[4]),
                derivate_fn(depth(base_width * 32), depth(base_width * 32), ratio=ratios[4])
            ))
        ]))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate, inplace=True),
            nn.Linear(depth(base_width * 32), num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _gssdnet_v1(
    depth_multiplier: float = 1.0,
    derivate_fn: nn.Module = Order2Block,
    downsample_fn: nn.Module = GaussianBlurBlock,
    lowpass_fn: nn.Module = LowPassFilterBlock,
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    model = GSSDNet(
        derivate_fn=derivate_fn,
        downsample_fn=downsample_fn,
        lowpass_fn=lowpass_fn,
        depth_multiplier=depth_multiplier,
        **kwargs
    )

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
def gsssdnet_o2_x1_0(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _gssdnet_v1(1.0, Order2Block, GaussianBlurBlock, LowPassFilterBlock, pretrained, pth, progress, **kwargs)


@export
def gsssdnet_o1_x1_0(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _gssdnet_v1(1.0, Order1Block, GaussianBlurBlock, LowPassFilterBlock, pretrained, pth, progress, **kwargs)


@export
def gsssdnet_o2_x0_75(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _gssdnet_v1(0.75, Order2Block, GaussianBlurBlock, LowPassFilterBlock, pretrained, pth, progress, **kwargs)


@export
def gsssdnet_o1_x0_75(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _gssdnet_v1(0.75, Order1Block, GaussianBlurBlock, LowPassFilterBlock, pretrained, pth, progress, **kwargs)


@export
def gsssdnet_o2_x0_5(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _gssdnet_v1(0.5, Order2Block, GaussianBlurBlock, LowPassFilterBlock, pretrained, pth, progress, **kwargs)


@export
def gsssdnet_o1_x0_5(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _gssdnet_v1(0.5, Order1Block, GaussianBlurBlock, LowPassFilterBlock, pretrained, pth, progress, **kwargs)
