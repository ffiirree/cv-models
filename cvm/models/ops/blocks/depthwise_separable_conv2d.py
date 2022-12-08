from torch import nn
from . import factory


class DepthwiseConv2d(nn.Conv2d):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        bias: bool = False,
    ):
        if padding is None:
            padding = ((kernel_size - 1) * (dilation - 1) + kernel_size) // 2

        super().__init__(
            inp, oup, kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=bias, groups=inp
        )


class PointwiseConv2d(nn.Conv2d):
    def __init__(
        self,
        inp,
        oup,
        stride: int = 1,
        bias: bool = False,
        groups: int = 1
    ):
        super().__init__(inp, oup, 1, stride=stride, padding=0, bias=bias, groups=groups)


class DepthwiseConv2dBN(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        normalizer_fn: nn.Module = None
    ):
        normalizer_fn = normalizer_fn or factory._NORMALIZER

        super().__init__(
            DepthwiseConv2d(inp, oup, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )

        if normalizer_fn:
            self.add_module(str(self.__len__()), normalizer_fn(oup))


class PointwiseConv2dBN(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        stride: int = 1,
        normalizer_fn: nn.Module = None
    ):
        normalizer_fn = normalizer_fn or factory._NORMALIZER

        super().__init__(
            PointwiseConv2d(inp, oup, stride=stride)
        )

        if normalizer_fn:
            self.add_module(str(self.__len__()), normalizer_fn(oup))


class DepthwiseBlock(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        normalizer_fn: nn.Module = None,
        activation_fn: nn.Module = None,
        norm_position: str = None
    ):
        super().__init__(
            DepthwiseConv2d(inp, oup, kernel_size, stride, padding=padding, dilation=dilation),
            *factory.norm_activation(oup, normalizer_fn, activation_fn, norm_position)
        )


class PointwiseBlock(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        stride: int = 1,
        groups: int = 1,
        normalizer_fn: nn.Module = None,
        activation_fn: nn.Module = None,
        norm_position: str = None,
    ):
        super().__init__(
            PointwiseConv2d(inp, oup, stride=stride, groups=groups),
            *factory.norm_activation(oup, normalizer_fn, activation_fn, norm_position)
        )
