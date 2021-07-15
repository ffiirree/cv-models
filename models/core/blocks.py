from typing import OrderedDict
import torch
import torch.nn as nn
from torch.nn.functional import group_norm


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
        bn_momentum: float = 0.1,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super().__init__(
            Conv2d3x3(in_channels, out_channels, stride=stride,
                      padding=padding, bias=bias, groups=groups),
            norm_layer(out_channels, momentum=bn_momentum)
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
        bn_momentum: float = 0.1,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super().__init__(
            Conv2d1x1(in_channels, out_channels, stride=stride,
                      padding=padding, bias=bias, groups=groups),
            norm_layer(out_channels, momentum=bn_momentum)
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
        bn_momentum: float = 0.1,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation_layer: nn.Module = nn.ReLU
    ):
        super().__init__(
            Conv2d1x1(in_channels, out_channels, stride=stride,
                      padding=padding, bias=bias, groups=groups),
            norm_layer(out_channels, momentum=bn_momentum),
            activation_layer(inplace=True)
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
        bn_momentum: float = 0.1,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation_layer: nn.Module = nn.ReLU
    ):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      bias=False, stride=stride, padding=padding, groups=groups),
            norm_layer(out_channels, momentum=bn_momentum),
            activation_layer(inplace=True),
        )


class ResBasicBlock(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        stride: int = 1,
        bn_momentum: float = 0.1,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation_layer: nn.Module = nn.ReLU
    ):
        super().__init__()

        self.branch1 = nn.Sequential(
            Conv2d3x3BN(inp, oup, stride=stride,
                        bn_momentum=bn_momentum, norm_layer=norm_layer),
            activation_layer(inplace=True),
            Conv2d3x3BN(oup, oup, bn_momentum=bn_momentum,
                        norm_layer=norm_layer)
        )

        self.branch2 = nn.Identity()

        if inp != oup or stride != 1:
            self.branch2 = Conv2d1x1BN(
                inp, oup, stride=stride, bn_momentum=bn_momentum, norm_layer=norm_layer)

        self.combine = Combine('ADD')
        self.relu = activation_layer(inplace=True)

    def forward(self, x):
        x = self.combine(self.branch1(x), self.branch2(x))
        x = self.relu(x)
        return x


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int = 1,
        bn_momentum: float = 0.1,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation_layer: nn.Module = nn.ReLU
    ):
        super().__init__()

        self.branch1 = nn.Sequential(
            Conv2d1x1BN(inp, oup, bn_momentum=bn_momentum,
                        norm_layer=norm_layer),
            activation_layer(inplace=True),
            Conv2d3x3BN(oup, oup, stride=stride,
                        bn_momentum=bn_momentum, norm_layer=norm_layer),
            activation_layer(inplace=True),
            Conv2d1x1BN(oup, oup * self.expansion,
                        bn_momentum=bn_momentum, norm_layer=norm_layer),
        )

        self.branch2 = nn.Identity()

        if stride != 1 or inp != oup * self.expansion:
            self.branch2 = Conv2d1x1BN(
                inp, oup * self.expansion, stride=stride, bn_momentum=bn_momentum, norm_layer=norm_layer)

        self.combine = Combine('ADD')
        self.relu = activation_layer(inplace=True)

    def forward(self, x):

        x = self.combine(self.branch1(x), self.branch2(x))
        x = self.relu(x)
        return x


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)
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
        stride: int = 1
    ):
        super().__init__(inp, oup, 1, stride=stride, padding=0, bias=False)


class DepthwiseConv2dBN(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bn_momentum: float = 0.1,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super().__init__(
            DepthwiseConv2d(inp, oup, kernel_size,
                            stride=stride, padding=padding),
            norm_layer(oup, momentum=bn_momentum)
        )


class PointwiseConv2dBN(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        stride: int = 1,
        bn_momentum: float = 0.1,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super().__init__(
            PointwiseConv2d(inp, oup, stride=stride),
            norm_layer(oup, momentum=bn_momentum)
        )


class DepthwiseBlock(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bn_momentum: float = 0.1,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation_layer: nn.Module = nn.ReLU
    ):
        super().__init__(
            DepthwiseConv2d(inp, oup, kernel_size,
                            stride=stride, padding=padding),
            norm_layer(oup, momentum=bn_momentum),
            activation_layer(inplace=True),
        )


class PointwiseBlock(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        stride: int = 1,
        bn_momentum: float = 0.1,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation_layer: nn.Module = nn.ReLU
    ):
        super().__init__(
            PointwiseConv2d(inp, oup, stride=stride),
            norm_layer(oup, momentum=bn_momentum),
            activation_layer(inplace=True),
        )


class SEBlock(nn.Module):
    def __init__(self, channels, ratio):
        super().__init__()

        squeezed_channels = max(1, int(channels * ratio))

        self.se = nn.Sequential(OrderedDict([
            ('pooling', nn.AdaptiveAvgPool2d((1, 1))),
            ('reduce', Conv2d1x1(channels, squeezed_channels, bias=True)),
            ('relu', nn.ReLU(inplace=True)),
            ('expand', Conv2d1x1(squeezed_channels, channels, bias=True)),
            ('sigmoid', nn.Sigmoid()),
        ]))

    def forward(self, x):
        return x * self.se(x)


class ChannelSplit(nn.Module):
    def __init__(self, groups: int):
        super().__init__()

        self.groups = groups

    def forward(self, x):
        return torch.chunk(x, self.groups, dim=1)

    def extra_repr(self):
        return f'groups={self.groups}'


class Combine(nn.Module):
    def __init__(self, method: str = 'ADD', *args, **kwargs):
        super().__init__()
        assert method in ['ADD', 'CONCAT'], ''

        self.method = method
        self._combine = self._add if self.method == 'ADD' else self._cat

    @staticmethod
    def _add(x1, x2):
        return x1 + x2

    @staticmethod
    def _cat(x1, x2):
        return torch.cat([x1, x2], dim=1)

    def forward(self, x1, x2):
        return self._combine(x1, x2)

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
        survival_prob: float = None,
        bn_momentum: float = 0.1,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation_layer: nn.Module = nn.ReLU
    ):
        super().__init__()

        self.inp = inp
        self.oup = oup
        self.stride = stride
        self.padding = padding if padding is not None else (kernel_size // 2)
        self.apply_residual = (self.stride == 1) and (self.inp == self.oup)

        layers = []
        if t != 1:
            layers.append(Conv2d1x1Block(
                inp, inp * t, bn_momentum=bn_momentum, norm_layer=norm_layer,
                activation_layer=activation_layer))

        layers.append(DepthwiseBlock(inp * t, inp * t, kernel_size, stride=self.stride, padding=self.padding,
                                     bn_momentum=bn_momentum, norm_layer=norm_layer, activation_layer=activation_layer))

        if (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1):
            layers.append(SEBlock(inp * t, se_ratio / t))

        layers.append(Conv2d1x1BN(
            inp * t, oup, bn_momentum=bn_momentum, norm_layer=norm_layer))

        if self.apply_residual and survival_prob:
            layers.append(DropBlock(survival_prob))

        self.branch1 = nn.Sequential(*layers)
        self.branch2 = nn.Identity() if self.apply_residual else None
        self.combine = Combine('ADD') if self.apply_residual else None

    def forward(self, x):
        if self.apply_residual:
            return self.combine(self.branch2(x), self.branch1(x))
        else:
            return self.branch1(x)
