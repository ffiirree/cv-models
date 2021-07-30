from typing import List, OrderedDict
from contextlib import contextmanager
import torch
import torch.nn as nn
from .functional import *

_BN_MOMENTUM: float = 0.1
_BN_EPSILON: float = 1e-5
_BN_POSIITON: str = 'before'
_NONLINEAR: nn.Module = nn.ReLU


@contextmanager
def batchnorm(
    momentum: float = _BN_MOMENTUM,
    eps: float = _BN_EPSILON,
    position: str = _BN_POSIITON
):
    global _BN_MOMENTUM, _BN_EPSILON, _BN_POSIITON

    _pre_momentum = _BN_MOMENTUM
    _pre_eps = _BN_EPSILON
    _pre_position = _BN_POSIITON

    _BN_MOMENTUM = momentum
    _BN_EPSILON = eps
    _BN_POSIITON = position

    yield

    _BN_MOMENTUM = _pre_momentum
    _BN_EPSILON = _pre_eps
    _BN_POSIITON = _pre_position


@contextmanager
def nonlinear(layer: nn.Module):
    global _NONLINEAR

    _pre_layer = _NONLINEAR
    _NONLINEAR = layer
    yield
    _NONLINEAR = _pre_layer


def norm_activation(
    channels,
    bn_epsilon: float = None,
    bn_momentum: float = None,
    norm_layer: nn.Module = nn.BatchNorm2d,
    activation_layer: nn.Module = None,
    norm_position: str = None
) -> List[nn.Module]:
    norm_position = norm_position if norm_position else _BN_POSIITON
    assert norm_position in ['before', 'after', 'none'], ''

    bn_epsilon = bn_epsilon if bn_epsilon else _BN_EPSILON
    bn_momentum = bn_momentum if bn_momentum else _BN_MOMENTUM
    activation_layer = activation_layer if activation_layer else _NONLINEAR

    layers = [
        norm_layer(channels, eps=bn_epsilon, momentum=bn_momentum),
        activation_layer(inplace=True)
    ]

    if norm_position == 'before':
        return layers
    elif norm_position == 'after':
        layers.reverse()
        return layers

    return [activation_layer(inplace=True)]


class Conv2d3x3(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        groups: int = 1
    ):
        super().__init__(
            in_channels, out_channels, 3, stride=stride,
            padding=padding, bias=bias, groups=groups
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
        padding: int = 1,
        bias: bool = False,
        groups: int = 1,
        bn_epsilon: float = None,
        bn_momentum: float = None,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        bn_epsilon = bn_epsilon if bn_epsilon else _BN_EPSILON
        bn_momentum = bn_momentum if bn_momentum else _BN_MOMENTUM

        super().__init__(
            Conv2d3x3(in_channels, out_channels, stride=stride,
                      padding=padding, bias=bias, groups=groups),
            norm_layer(out_channels, eps=bn_epsilon, momentum=bn_momentum)
        )


class Conv2d1x1BN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        groups: int = 1,
        bn_epsilon: float = None,
        bn_momentum: float = None,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        bn_epsilon = bn_epsilon if bn_epsilon else _BN_EPSILON
        bn_momentum = bn_momentum if bn_momentum else _BN_MOMENTUM

        super().__init__(
            Conv2d1x1(in_channels, out_channels, stride=stride,
                      padding=padding, bias=bias, groups=groups),
            norm_layer(out_channels, eps=bn_epsilon, momentum=bn_momentum)
        )


class Conv2d1x1Block(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        groups: int = 1,
        bn_epsilon: float = None,
        bn_momentum: float = None,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation_layer: nn.Module = None,
        norm_position: str = None
    ):
        super().__init__(
            Conv2d1x1(in_channels, out_channels, stride=stride,
                      padding=padding, bias=bias, groups=groups),
            *norm_activation(out_channels, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum,
                             norm_layer=norm_layer, activation_layer=activation_layer,
                             norm_position=norm_position)
        )


class Conv2dBlock(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        bn_epsilon: float = None,
        bn_momentum: float = None,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation_layer: nn.Module = None,
        norm_position: str = None,
    ):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      bias=False, stride=stride, padding=padding, groups=groups),
            *norm_activation(out_channels, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum,
                             norm_layer=norm_layer, activation_layer=activation_layer,
                             norm_position=norm_position)
        )


class ResBasicBlock(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        stride: int = 1,
        bn_epsilon: float = None,
        bn_momentum: float = None,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation_layer: nn.Module = None
    ):
        super().__init__()

        bn_epsilon = bn_epsilon if bn_epsilon else _BN_EPSILON
        bn_momentum = bn_momentum if bn_momentum else _BN_MOMENTUM
        activation_layer = activation_layer if activation_layer else _NONLINEAR

        self.branch1 = nn.Sequential(
            Conv2d3x3BN(inp, oup, stride=stride,
                        bn_momentum=bn_momentum, norm_layer=norm_layer),
            activation_layer(inplace=True),
            Conv2d3x3BN(oup, oup, eps=bn_epsilon, bn_momentum=bn_momentum,
                        norm_layer=norm_layer)
        )

        self.branch2 = nn.Identity()

        if inp != oup or stride != 1:
            self.branch2 = Conv2d1x1BN(
                inp, oup, stride=stride, bn_momentum=bn_momentum, norm_layer=norm_layer)

        self.combine = Combine('ADD')
        self.relu = activation_layer(inplace=True)

    def forward(self, x):
        x = self.combine([self.branch1(x), self.branch2(x)])
        x = self.relu(x)
        return x


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int = 1,
        bn_epsilon: float = None,
        bn_momentum: float = None,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation_layer: nn.Module = None
    ):
        super().__init__()

        bn_epsilon = bn_epsilon if bn_epsilon else _BN_EPSILON
        bn_momentum = bn_momentum if bn_momentum else _BN_MOMENTUM
        activation_layer = activation_layer if activation_layer else _NONLINEAR

        self.branch1 = nn.Sequential(
            Conv2d1x1BN(inp, oup, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum,
                        norm_layer=norm_layer),
            activation_layer(inplace=True),
            Conv2d3x3BN(oup, oup, stride=stride,
                        bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, norm_layer=norm_layer),
            activation_layer(inplace=True),
            Conv2d1x1BN(oup, oup * self.expansion,
                        bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, norm_layer=norm_layer),
        )

        self.branch2 = nn.Identity()

        if stride != 1 or inp != oup * self.expansion:
            self.branch2 = Conv2d1x1BN(
                inp, oup * self.expansion, stride=stride, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, norm_layer=norm_layer)

        self.combine = Combine('ADD')
        self.relu = activation_layer(inplace=True)

    def forward(self, x):

        x = self.combine([self.branch1(x), self.branch2(x)])
        x = self.relu(x)
        return x


class ChannelShuffle(nn.Module):
    def __init__(self, groups: int):
        super().__init__()

        self.groups = groups

    def forward(self, x):
        return channel_shuffle(x, self.groups)

    def extra_repr(self):
        return 'groups={}'.format(self.groups)


class DepthwiseConv2d(nn.Conv2d):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__(
            inp, oup, kernel_size, stride=stride, padding=padding, bias=False, groups=inp
        )


class PointwiseConv2d(nn.Conv2d):
    def __init__(
        self,
        inp,
        oup,
        stride: int = 1,
        groups: int = 1
    ):
        super().__init__(inp, oup, 1, stride=stride, padding=0, bias=False, groups=groups)


class DepthwiseConv2dBN(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bn_epsilon: float = None,
        bn_momentum: float = None,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        bn_epsilon = bn_epsilon if bn_epsilon else _BN_EPSILON
        bn_momentum = bn_momentum if bn_momentum else _BN_MOMENTUM

        super().__init__(
            DepthwiseConv2d(inp, oup, kernel_size,
                            stride=stride, padding=padding),
            norm_layer(oup, eps=bn_epsilon, momentum=bn_momentum)
        )


class PointwiseConv2dBN(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        stride: int = 1,
        bn_epsilon: float = None,
        bn_momentum: float = None,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        bn_epsilon = bn_epsilon if bn_epsilon else _BN_EPSILON
        bn_momentum = bn_momentum if bn_momentum else _BN_MOMENTUM

        super().__init__(
            PointwiseConv2d(inp, oup, stride=stride),
            norm_layer(oup, eps=bn_epsilon, momentum=bn_momentum)
        )


class DepthwiseBlock(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bn_epsilon: float = None,
        bn_momentum: float = None,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation_layer: nn.Module = None,
        norm_position: str = None
    ):
        super().__init__(
            DepthwiseConv2d(inp, oup, kernel_size,
                            stride=stride, padding=padding),
            *norm_activation(oup, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum,
                             norm_layer=norm_layer, activation_layer=activation_layer,
                             norm_position=norm_position)
        )


class PointwiseBlock(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        stride: int = 1,
        groups: int = 1,
        bn_epsilon: float = None,
        bn_momentum: float = None,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation_layer: nn.Module = None,
        norm_position: str = None,
    ):
        super().__init__(
            PointwiseConv2d(inp, oup, stride=stride, groups=groups),
            *norm_activation(oup, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum,
                             norm_layer=norm_layer, activation_layer=activation_layer,
                             norm_position=norm_position)
        )


class SEBlock(nn.Module):
    def __init__(self, channels, ratio):
        super().__init__()

        squeezed_channels = make_divisible(int(channels * ratio), 8)

        self.se = nn.Sequential(OrderedDict([
            ('pooling', nn.AdaptiveAvgPool2d((1, 1))),
            ('reduce', Conv2d1x1(channels, squeezed_channels, bias=True)),
            ('relu', nn.ReLU(inplace=True)),
            ('expand', Conv2d1x1(squeezed_channels, channels, bias=True)),
            ('sigmoid', nn.Sigmoid()),
        ]))

    def forward(self, x):
        return x * self.se(x)


class ChannelChunk(nn.Module):
    def __init__(self, groups: int):
        super().__init__()

        self.groups = groups

    def forward(self, x):
        return torch.chunk(x, self.groups, dim=1)

    def extra_repr(self):
        return f'groups={self.groups}'


class ChannelSplit(nn.Module):
    def __init__(self, sections):
        super().__init__()

        self.sections = sections

    def forward(self, x):
        return torch.split(x, self.sections, dim=1)

    def extra_repr(self):
        return f'sections={self.sections}'


class Combine(nn.Module):
    def __init__(self, method: str = 'ADD', *args, **kwargs):
        super().__init__()
        assert method in ['ADD', 'CONCAT'], ''

        self.method = method
        self._combine = self._add if self.method == 'ADD' else self._cat

    @staticmethod
    def _add(x):
        y = 0
        for xi in x:
            y += xi
        return y

    @staticmethod
    def _cat(x):
        return torch.cat(x, dim=1)

    def forward(self, x):
        return self._combine(x)

    def extra_repr(self):
        return f'method=\'{self.method}\''


class DropBlock(nn.Module):
    def __init__(self, survival_prob: float):
        super().__init__()

        self.p = survival_prob

    def forward(self, x):
        if not self.training:
            return x

        probs = self.p + \
            torch.rand([x.size()[0], 1, 1, 1], dtype=x.dtype, device=x.device)
        return (x / self.p) * probs.floor_()

    def extra_repr(self):
        return f'survival_prob={self.p}'


class InvertedResidualBlock(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        t,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        se_ratio: float = None,
        se_ind: bool = False,
        survival_prob: float = None,
        bn_epsilon: float = None,
        bn_momentum: float = None,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation_layer: nn.Module = None
    ):
        super().__init__()

        self.inp = inp
        self.planes = int(self.inp * t)
        self.oup = oup
        self.stride = stride
        self.padding = padding if padding is not None else (kernel_size // 2)
        self.apply_residual = (self.stride == 1) and (self.inp == self.oup)
        self.se_ratio = se_ratio if se_ind or se_ratio is None else (
            se_ratio / t)
        self.has_se = (self.se_ratio is not None) and (
            self.se_ratio > 0) and (self.se_ratio <= 1)
        bn_epsilon = bn_epsilon if bn_epsilon else _BN_EPSILON
        bn_momentum = bn_momentum if bn_momentum else _BN_MOMENTUM
        activation_layer = activation_layer if activation_layer else _NONLINEAR

        layers = []
        if t != 1:
            layers.append(Conv2d1x1Block(
                inp, self.planes,
                bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, norm_layer=norm_layer,
                activation_layer=activation_layer))

        layers.append(DepthwiseBlock(self.planes, self.planes, kernel_size, stride=self.stride, padding=self.padding,
                                     bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, norm_layer=norm_layer, activation_layer=activation_layer))

        if self.has_se:
            layers.append(SEBlock(self.planes, self.se_ratio))

        layers.append(Conv2d1x1BN(
            self.planes, oup, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, norm_layer=norm_layer))

        if self.apply_residual and survival_prob:
            layers.append(DropBlock(survival_prob))

        self.branch1 = nn.Sequential(*layers)
        self.branch2 = nn.Identity() if self.apply_residual else None
        self.combine = Combine('ADD') if self.apply_residual else None

    def forward(self, x):
        if self.apply_residual:
            return self.combine([self.branch2(x), self.branch1(x)])
        else:
            return self.branch1(x)


class FusedInvertedResidualBlock(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        t,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        se_ratio: float = None,
        se_ind: bool = False,
        survival_prob: float = None,
        bn_epsilon: float = None,
        bn_momentum: float = None,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation_layer: nn.Module = None
    ):
        super().__init__()

        self.inp = inp
        self.planes = int(self.inp * t)
        self.oup = oup
        self.stride = stride
        self.padding = padding if padding is not None else (kernel_size // 2)
        self.apply_residual = (self.stride == 1) and (self.inp == self.oup)
        self.se_ratio = se_ratio if se_ind or se_ratio is None else (
            se_ratio / t)
        self.has_se = (self.se_ratio is not None) and (
            self.se_ratio > 0) and (self.se_ratio <= 1)
        bn_epsilon = bn_epsilon if bn_epsilon else _BN_EPSILON
        bn_momentum = bn_momentum if bn_momentum else _BN_MOMENTUM
        activation_layer = activation_layer if activation_layer else _NONLINEAR

        layers = [
            Conv2dBlock(inp, self.planes, kernel_size, stride=self.stride, padding=self.padding,
                        bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, norm_layer=norm_layer, activation_layer=activation_layer)
        ]

        if self.has_se:
            layers.append(SEBlock(self.planes, self.se_ratio))

        layers.append(Conv2d1x1BN(
            self.planes, oup, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, norm_layer=norm_layer))

        if self.apply_residual and survival_prob:
            layers.append(DropBlock(survival_prob))

        self.branch1 = nn.Sequential(*layers)
        self.branch2 = nn.Identity() if self.apply_residual else None
        self.combine = Combine('ADD') if self.apply_residual else None

    def forward(self, x):
        if self.apply_residual:
            return self.combine([self.branch2(x), self.branch1(x)])
        else:
            return self.branch1(x)


class SplitIdentityDepthWiseConv2dLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.channels = channels // 2

        self.split = ChannelChunk(2)
        self.branch1 = nn.Identity()
        self.branch2 = DepthwiseConv2d(self.channels, self.channels)
        self.combine = Combine('CONCAT')

    def forward(self, x):
        x1, x2 = self.split(x)
        out = self.combine([self.branch1(x1), self.branch2(x2)])
        return out


class MuxDepthwiseConv2d(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        mux_layer: nn.Module = None
    ):
        super().__init__()

        self.channels = channels
        self.mux_layer = mux_layer

        if self.mux_layer is not None:
            self.channels = channels // 2

        self.layer = DepthwiseConv2d(
            self.channels, self.channels, kernel_size, stride, padding)

    def forward(self, x):
        if self.mux_layer is not None:
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1 = self.layer(x1)
            x2 = self.mux_layer(x2)
            return torch.cat([x1, x2], dim=1)
        else:
            return self.layer(x)


class SharedDepthwiseConv2d(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        t: int = 2
    ):
        super().__init__()

        self.channels = channels // t
        self.t = t

        self.dw = DepthwiseConv2d(
            self.channels, self.channels, kernel_size, stride, padding)

    def forward(self, x):
        x = torch.chunk(x, self.t, dim=1)
        x = [self.dw(xi) for xi in x]
        return torch.cat(x, dim=1)
