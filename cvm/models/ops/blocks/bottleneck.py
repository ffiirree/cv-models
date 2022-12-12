import torch
import torch.nn as nn

from . import factory
from .squeeze_excite import SEBlock
from .drop import StochasticDepth
from .vanilla_conv2d import Conv2d3x3, Conv2d1x1
from .channel import Combine

from typing import OrderedDict


class ResBasicBlockV1(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inp,
        oup,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        width_per_group: int = 64,
        rd_ratio: float = None,
        drop_path_rate: float = None,
        use_resnetd_shortcut: bool = False,
        normalizer_fn: nn.Module = None,
        activation_fn: nn.Module = None
    ):
        super().__init__()

        normalizer_fn = normalizer_fn or factory._NORMALIZER
        activation_fn = activation_fn or factory._ACTIVATION

        self.has_attn = rd_ratio is not None and rd_ratio > 0 and rd_ratio <= 1
        self.use_shortcut = stride != 1 or inp != oup * self.expansion

        if width_per_group != 64:
            raise ValueError('width_per_group are not supported!')

        self.branch1 = nn.Sequential(OrderedDict([
            ('conv1', Conv2d3x3(inp, oup, stride=stride, dilation=dilation, groups=groups)),
            ('norm1', normalizer_fn(oup)),
            ('relu1', activation_fn()),
            ('conv2', Conv2d3x3(oup, oup, dilation=dilation)),
            ('norm2', normalizer_fn(oup))
        ]))

        if self.has_attn:
            self.branch1.add_module('se', SEBlock(oup, rd_ratio=rd_ratio))

        if drop_path_rate:
            self.branch1.add_module('drop', StochasticDepth(1. - drop_path_rate))

        self.branch2 = nn.Identity()

        if self.use_shortcut:
            if use_resnetd_shortcut and stride != 1:
                self.branch2 = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, stride=stride)),
                    ('conv', Conv2d1x1(inp, oup)),
                    ('norm', normalizer_fn(oup))
                ]))
            else:
                self.branch2 = nn.Sequential(OrderedDict([
                    ('conv', Conv2d1x1(inp, oup, stride=stride)),
                    ('norm', normalizer_fn(oup))
                ]))

        self.combine = Combine('ADD')
        self.relu = activation_fn()

    def zero_init_last_bn(self):
        nn.init.zeros_(self.branch1.norm2.weight)

    def forward(self, x):
        x = self.combine([self.branch1(x), self.branch2(x)])
        x = self.relu(x)
        return x


class BottleneckV1(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        width_per_group: int = 64,
        rd_ratio: float = None,
        drop_path_rate: float = None,
        use_resnetd_shortcut: bool = False,
        normalizer_fn: nn.Module = None,
        activation_fn: nn.Module = None
    ):
        super().__init__()

        normalizer_fn = normalizer_fn or factory._NORMALIZER
        activation_fn = activation_fn or factory._ACTIVATION

        width = int(oup * (width_per_group / 64)) * groups

        self.has_attn = rd_ratio is not None and rd_ratio > 0 and rd_ratio <= 1
        self.use_shortcut = stride != 1 or inp != oup * self.expansion

        self.branch1 = nn.Sequential(OrderedDict([
            ('conv1', Conv2d1x1(inp, width)),
            ('norm1', normalizer_fn(width)),
            ('relu1', activation_fn()),
            ('conv2', Conv2d3x3(width, width, stride=stride, dilation=dilation, groups=groups)),
            ('norm2', normalizer_fn(width)),
            ('relu2', activation_fn()),
            ('conv3', Conv2d1x1(width, oup * self.expansion)),
            ('norm3', normalizer_fn(oup * self.expansion,))
        ]))

        if self.has_attn:
            self.branch1.add_module('se', SEBlock(oup * self.expansion, rd_ratio=rd_ratio))

        if drop_path_rate:
            self.branch1.add_module('drop', StochasticDepth(1. - drop_path_rate))

        self.branch2 = nn.Identity()

        if self.use_shortcut:
            if use_resnetd_shortcut and stride != 1:
                self.branch2 = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, stride=stride)),
                    ('conv', Conv2d1x1(inp, oup * self.expansion)),
                    ('norm', normalizer_fn(oup * self.expansion))
                ]))
            else:
                self.branch2 = nn.Sequential(OrderedDict([
                    ('conv', Conv2d1x1(inp, oup * self.expansion, stride=stride)),
                    ('norm', normalizer_fn(oup * self.expansion))
                ]))

        self.combine = Combine('ADD')
        self.relu = activation_fn()

    def zero_init_last_bn(self):
        nn.init.zeros_(self.branch1.norm3.weight)

    def forward(self, x):
        x = self.combine([self.branch1(x), self.branch2(x)])
        x = self.relu(x)
        return x


class ResBasicBlockV2(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inp,
        oup,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        width_per_group: int = 64,
        rd_ratio: float = None,
        drop_path_rate: float = None,
        use_resnetd_shortcut: bool = False,
        normalizer_fn: nn.Module = None,
        activation_fn: nn.Module = None
    ):
        super().__init__()

        normalizer_fn = normalizer_fn or factory._NORMALIZER
        activation_fn = activation_fn or factory._ACTIVATION

        self.has_attn = rd_ratio is not None and rd_ratio > 0 and rd_ratio <= 1
        self.use_shortcut = stride != 1 or inp != oup

        if width_per_group != 64:
            raise ValueError('width_per_group are not supported!')

        self.branch1 = nn.Sequential(OrderedDict([
            ('norm1', normalizer_fn(inp)),
            ('relu1', activation_fn()),
            ('conv1', Conv2d3x3(inp, oup, stride=stride, dilation=dilation, groups=groups)),
            ('norm2', normalizer_fn(oup)),
            ('relu2', activation_fn()),
            ('conv2', Conv2d3x3(oup, oup))
        ]))

        if self.has_attn:
            self.branch1.add_module('se', SEBlock(oup, rd_ratio=rd_ratio))

        if drop_path_rate:
            self.branch1.add_module('drop', StochasticDepth(1. - drop_path_rate))

        self.branch2 = nn.Identity()

        if self.use_shortcut:
            self.branch2 = nn.Sequential()
            if use_resnetd_shortcut and stride != 1:
                self.branch2.add_module('pool', nn.AvgPool2d(2, stride))
                stride = 1

            self.branch2.add_module('norm', normalizer_fn(inp))
            self.branch2.add_module('relu', activation_fn())
            self.branch2.add_module('conv', Conv2d1x1(inp, oup, stride))

        self.combine = Combine('ADD')

    def zero_init_last_bn(self):
        nn.init.zeros_(self.branch1.norm2.weight)

    def forward(self, x):
        x = self.combine([self.branch1(x), self.branch2(x)])
        return x


class BottleneckV2(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        width_per_group: int = 64,
        rd_ratio: float = None,
        drop_path_rate: float = None,
        use_resnetd_shortcut: bool = False,
        normalizer_fn: nn.Module = None,
        activation_fn: nn.Module = None
    ):
        super().__init__()

        normalizer_fn = normalizer_fn or factory._NORMALIZER
        activation_fn = activation_fn or factory._ACTIVATION

        width = int(oup * (width_per_group / 64)) * groups

        self.has_attn = rd_ratio is not None and rd_ratio > 0 and rd_ratio <= 1
        self.use_shortcut = stride != 1 or inp != oup * self.expansion

        self.branch1 = nn.Sequential(OrderedDict([
            ('norm1', normalizer_fn(inp)),
            ('relu1', activation_fn()),
            ('conv1', Conv2d1x1(inp, width)),
            ('norm2', normalizer_fn(width)),
            ('relu2', activation_fn()),
            ('conv2', Conv2d3x3(width, width, stride=stride, dilation=dilation, groups=groups)),
            ('norm3', normalizer_fn(width)),
            ('relu3', activation_fn()),
            ('conv3', Conv2d1x1(width, oup * self.expansion))
        ]))

        if self.has_attn:
            self.branch1.add_module('se', SEBlock(
                oup * self.expansion, rd_ratio=rd_ratio))

        if drop_path_rate:
            self.branch1.add_module('drop', StochasticDepth(1. - drop_path_rate))

        self.branch2 = nn.Identity()

        if self.use_shortcut:
            self.branch2 = nn.Sequential()
            if use_resnetd_shortcut and stride != 1:
                self.branch2.add_module('pool', nn.AvgPool2d(2, stride))
                stride = 1

            self.branch2.add_module('norm', normalizer_fn(inp))
            self.branch2.add_module('relu', activation_fn())
            self.branch2.add_module('conv', Conv2d1x1(inp, oup * self.expansion, stride))

        self.combine = Combine('ADD')

    def zero_init_last_bn(self):
        nn.init.zeros_(self.branch1.norm3.weight)

    def forward(self, x):
        x = self.combine([self.branch1(x), self.branch2(x)])
        return x
