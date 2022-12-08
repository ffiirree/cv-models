from torch import nn
from . import factory


class Conv2d3x3(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        bias: bool = False,
        groups: int = 1
    ):
        padding = padding if padding is not None else dilation
        super().__init__(
            in_channels, out_channels, 3, stride=stride,
            padding=padding, dilation=dilation, bias=bias, groups=groups
        )


class Conv2d1x1(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        groups: int = 1
    ):
        super().__init__(
            in_channels, out_channels, 1, stride=stride,
            padding=padding, bias=bias, groups=groups
        )


class Conv2d3x3BN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        bias: bool = False,
        groups: int = 1,
        normalizer_fn: nn.Module = None
    ):
        normalizer_fn = normalizer_fn or factory._NORMALIZER
        padding = padding if padding is not None else dilation

        super().__init__(
            Conv2d3x3(in_channels, out_channels, stride=stride,
                      padding=padding, dilation=dilation, bias=bias, groups=groups)
        )
        if normalizer_fn:
            self.add_module(str(self.__len__()), normalizer_fn(out_channels))


class Conv2d1x1BN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        groups: int = 1,
        normalizer_fn: nn.Module = None
    ):
        normalizer_fn = normalizer_fn or factory._NORMALIZER

        super().__init__(
            Conv2d1x1(in_channels, out_channels, stride=stride,
                      padding=padding, bias=bias, groups=groups)
        )
        if normalizer_fn:
            self.add_module(str(self.__len__()), normalizer_fn(out_channels))


class Conv2d1x1Block(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        groups: int = 1,
        normalizer_fn: nn.Module = None,
        activation_fn: nn.Module = None,
        norm_position: str = None
    ):
        super().__init__(
            Conv2d1x1(in_channels, out_channels, stride=stride,
                      padding=padding, bias=bias, groups=groups),
            *factory.norm_activation(out_channels, normalizer_fn, activation_fn, norm_position)
        )


class Conv2dBlock(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        bias: bool = False,
        groups: int = 1,
        normalizer_fn: nn.Module = None,
        activation_fn: nn.Module = None,
        norm_position: str = None,
    ):
        if padding is None:
            padding = ((kernel_size - 1) * (dilation - 1) + kernel_size) // 2

        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups),
            *factory.norm_activation(out_channels, normalizer_fn, activation_fn, norm_position)
        )
