from typing import List, OrderedDict
from contextlib import contextmanager
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from .functional import *

_NORM_POSIITON: str = 'before'
_NORMALIZER: nn.Module = nn.BatchNorm2d
_NONLINEAR: nn.Module = partial(nn.ReLU, inplace=True)
_SE_INNER_NONLINEAR: nn.Module = partial(nn.ReLU, inplace=True)
_SE_GATING_FN: nn.Module = nn.Sigmoid
_SE_DIVISOR: int = 8
_SE_USE_NORM: bool = False


class Nil:
    ...


@contextmanager
def normalizer(
    # _NORMALIZER can be None, Nil: _NORMALIZER->_NORMALIZER, None: _NORMALIZER->None
    fn: nn.Module = Nil,
    position: str = None
):

    global _NORMALIZER, _NORM_POSIITON

    fn = _NORMALIZER if fn == Nil else fn
    position = position or _NORM_POSIITON

    _pre_normalizer = _NORMALIZER
    _pre_position = _NORM_POSIITON

    _NORMALIZER = fn
    _NORM_POSIITON = position

    yield

    _NORMALIZER = _pre_normalizer
    _NORM_POSIITON = _pre_position


@contextmanager
def nonlinear(layer: nn.Module):
    global _NONLINEAR

    _pre_layer = _NONLINEAR
    _NONLINEAR = layer
    yield
    _NONLINEAR = _pre_layer


@contextmanager
def se(
    inner_nonlinear: nn.Module = _SE_INNER_NONLINEAR,
    gating_fn: nn.Module = _SE_GATING_FN,
    divisor: int = _SE_DIVISOR,
    use_norm: bool = _SE_USE_NORM
):
    global _SE_INNER_NONLINEAR
    global _SE_GATING_FN
    global _SE_DIVISOR
    global _SE_USE_NORM

    _pre_inner_fn = _SE_INNER_NONLINEAR
    _pre_fn = _SE_GATING_FN
    _pre_divisor = _SE_DIVISOR
    _pre_use_norm = _SE_USE_NORM
    _SE_INNER_NONLINEAR = inner_nonlinear
    _SE_GATING_FN = gating_fn
    _SE_DIVISOR = divisor
    _SE_USE_NORM = use_norm
    yield
    _SE_INNER_NONLINEAR = _pre_inner_fn
    _SE_GATING_FN = _pre_fn
    _SE_DIVISOR = _pre_divisor
    _SE_USE_NORM = _pre_use_norm


def normalizer_fn(channels):
    return _NORMALIZER(channels)


def activation_fn():
    return _NONLINEAR()


def norm_activation(
    channels,
    normalizer_fn: nn.Module = None,
    activation_fn: nn.Module = None,
    norm_position: str = None
) -> List[nn.Module]:
    norm_position = norm_position or _NORM_POSIITON
    assert norm_position in ['before', 'after', 'none'], ''

    normalizer_fn = normalizer_fn or _NORMALIZER
    activation_fn = activation_fn or _NONLINEAR

    if normalizer_fn == None and activation_fn == None:
        return []

    if normalizer_fn == None:
        return [activation_fn()]

    if activation_fn == None:
        return [normalizer_fn(channels)]

    if norm_position == 'after':
        return [activation_fn(), normalizer_fn(channels)]

    return [normalizer_fn(channels), activation_fn()]


class Stage(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        self.alpha = nn.Parameter(torch.ones(dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(dim, 1, 1))

    def forward(self, x):
        return self.alpha * x + self.beta

    def extra_repr(self):
        return f'{self.dim}'


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
        normalizer_fn = normalizer_fn or _NORMALIZER
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
        normalizer_fn = normalizer_fn or _NORMALIZER

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
            *norm_activation(out_channels, normalizer_fn, activation_fn, norm_position)
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
            *norm_activation(out_channels, normalizer_fn, activation_fn, norm_position)
        )


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
        se_ratio: float = None,
        drop_path_rate: float = None,
        use_resnetd_shortcut: bool = False,
        normalizer_fn: nn.Module = None,
        activation_fn: nn.Module = None
    ):
        super().__init__()

        normalizer_fn = normalizer_fn or _NORMALIZER
        activation_fn = activation_fn or _NONLINEAR

        self.has_se = se_ratio is not None and se_ratio > 0 and se_ratio <= 1
        self.use_shortcut = stride != 1 or inp != oup * self.expansion

        if width_per_group != 64:
            raise ValueError('width_per_group are not supported!')

        self.branch1 = nn.Sequential(OrderedDict([
            ('conv1', Conv2d3x3(
                inp, oup, stride=stride, dilation=dilation, groups=groups
            )),
            ('norm1', normalizer_fn(oup)),
            ('relu1', activation_fn()),
            ('conv2', Conv2d3x3(oup, oup, dilation=dilation)),
            ('norm2', normalizer_fn(oup))
        ]))

        if self.has_se:
            self.branch1.add_module('se', SEBlock(oup, se_ratio))

        if drop_path_rate:
            self.branch1.add_module('drop', DropPath(1. - drop_path_rate))

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
        se_ratio: float = None,
        drop_path_rate: float = None,
        use_resnetd_shortcut: bool = False,
        normalizer_fn: nn.Module = None,
        activation_fn: nn.Module = None
    ):
        super().__init__()

        normalizer_fn = normalizer_fn or _NORMALIZER
        activation_fn = activation_fn or _NONLINEAR

        width = int(oup * (width_per_group / 64)) * groups

        self.has_se = se_ratio is not None and se_ratio > 0 and se_ratio <= 1
        self.use_shortcut = stride != 1 or inp != oup * self.expansion

        self.branch1 = nn.Sequential(OrderedDict([
            ('conv1', Conv2d1x1(inp, width)),
            ('norm1', normalizer_fn(width)),
            ('relu1', activation_fn()),
            ('conv2', Conv2d3x3(
                width, width, stride=stride, dilation=dilation, groups=groups
            )),
            ('norm2', normalizer_fn(width)),
            ('relu2', activation_fn()),
            ('conv3', Conv2d1x1(width, oup * self.expansion)),
            ('norm3', normalizer_fn(oup * self.expansion,))
        ]))

        if self.has_se:
            self.branch1.add_module('se', SEBlock(
                oup * self.expansion, se_ratio))

        if drop_path_rate:
            self.branch1.add_module('drop', DropPath(1. - drop_path_rate))

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
        se_ratio: float = None,
        drop_path_rate: float = None,
        use_resnetd_shortcut: bool = False,
        normalizer_fn: nn.Module = None,
        activation_fn: nn.Module = None
    ):
        super().__init__()

        normalizer_fn = normalizer_fn or _NORMALIZER
        activation_fn = activation_fn or _NONLINEAR

        self.has_se = se_ratio is not None and se_ratio > 0 and se_ratio <= 1
        self.use_shortcut = stride != 1 or inp != oup

        if width_per_group != 64:
            raise ValueError('width_per_group are not supported!')

        self.branch1 = nn.Sequential(OrderedDict([
            ('norm1', normalizer_fn(inp)),
            ('relu1', activation_fn()),
            ('conv1', Conv2d3x3(
                inp, oup, stride=stride, dilation=dilation, groups=groups
            )),
            ('norm2', normalizer_fn(oup)),
            ('relu2', activation_fn()),
            ('conv2', Conv2d3x3(oup, oup))
        ]))

        if self.has_se:
            self.branch1.add_module('se', SEBlock(oup, se_ratio))

        if drop_path_rate:
            self.branch1.add_module('drop', DropPath(1. - drop_path_rate))

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
        se_ratio: float = None,
        drop_path_rate: float = None,
        use_resnetd_shortcut: bool = False,
        normalizer_fn: nn.Module = None,
        activation_fn: nn.Module = None
    ):
        super().__init__()

        normalizer_fn = normalizer_fn or _NORMALIZER
        activation_fn = activation_fn or _NONLINEAR

        width = int(oup * (width_per_group / 64)) * groups

        self.has_se = se_ratio is not None and se_ratio > 0 and se_ratio <= 1
        self.use_shortcut = stride != 1 or inp != oup * self.expansion

        self.branch1 = nn.Sequential(OrderedDict([
            ('norm1', normalizer_fn(inp)),
            ('relu1', activation_fn()),
            ('conv1', Conv2d1x1(inp, width)),
            ('norm2', normalizer_fn(width)),
            ('relu2', activation_fn()),
            ('conv2', Conv2d3x3(
                width, width, stride=stride, dilation=dilation, groups=groups
            )),
            ('norm3', normalizer_fn(width)),
            ('relu3', activation_fn()),
            ('conv3', Conv2d1x1(width, oup * self.expansion))
        ]))

        if self.has_se:
            self.branch1.add_module('se', SEBlock(
                oup * self.expansion, se_ratio))

        if drop_path_rate:
            self.branch1.add_module('drop', DropPath(1. - drop_path_rate))

        self.branch2 = nn.Identity()

        if self.use_shortcut:
            self.branch2 = nn.Sequential()
            if use_resnetd_shortcut and stride != 1:
                self.branch2.add_module('pool', nn.AvgPool2d(2, stride))
                stride = 1

            self.branch2.add_module('norm', normalizer_fn(inp))
            self.branch2.add_module('relu', activation_fn())
            self.branch2.add_module('conv', Conv2d1x1(
                inp, oup * self.expansion, stride))

        self.combine = Combine('ADD')

    def zero_init_last_bn(self):
        nn.init.zeros_(self.branch1.norm3.weight)

    def forward(self, x):
        x = self.combine([self.branch1(x), self.branch2(x)])
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
        normalizer_fn = normalizer_fn or _NORMALIZER

        super().__init__(
            DepthwiseConv2d(inp, oup, kernel_size, stride=stride,
                            padding=padding, dilation=dilation),
            normalizer_fn(oup)
        )


class PointwiseConv2dBN(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        stride: int = 1,
        normalizer_fn: nn.Module = None
    ):
        normalizer_fn = normalizer_fn or _NORMALIZER

        super().__init__(PointwiseConv2d(inp, oup, stride=stride))
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
            DepthwiseConv2d(inp, oup, kernel_size, stride,
                            padding=padding, dilation=dilation),
            *norm_activation(oup, normalizer_fn, activation_fn, norm_position)
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
            *norm_activation(oup, normalizer_fn, activation_fn, norm_position)
        )


class SEBlock(nn.Module):
    """Squeeze excite block
    """

    def __init__(
        self,
        channels,
        ratio,
        inner_activation_fn: nn.Module = partial(nn.ReLU, inplace=True),
        gating_fn: nn.Module = None
    ):
        super().__init__()

        squeezed_channels = make_divisible(int(channels * ratio), _SE_DIVISOR)
        gating_fn = gating_fn or _SE_GATING_FN

        layers = OrderedDict([])

        layers['pooling'] = nn.AdaptiveAvgPool2d((1, 1))
        layers['reduce'] = Conv2d1x1(channels, squeezed_channels, bias=True)
        if _SE_USE_NORM:
            layers['norm'] = _NORMALIZER(squeezed_channels)
        layers['relu'] = inner_activation_fn()
        layers['expand'] = Conv2d1x1(squeezed_channels, channels, bias=True)
        layers['sigmoid'] = gating_fn()

        self.se = nn.Sequential(layers)

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
        return x[0] + x[1]

    @staticmethod
    def _cat(x):
        return torch.cat(x, dim=1)

    def forward(self, x):
        return self._combine(x)

    def extra_repr(self):
        return f'method=\'{self.method}\''


class DropPath(nn.Module):
    """Stochastic Depth: Drop paths per sample (when applied in main path of residual blocks)"""

    def __init__(self, survival_prob: float):
        super().__init__()

        self.p = survival_prob

    def forward(self, x):
        if self.p == 1. or not self.training:
            return x

        # work with diff dim tensors, not just 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        probs = self.p + torch.rand(shape, dtype=x.dtype, device=x.device)
        # We therefore need to re-calibrate the outputs of any given function f
        # by the expected number of times it participates in training, p.
        return (x / self.p) * probs.floor_()

    def extra_repr(self):
        return f'survival_prob={self.p}'


class GaussianFilter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stride: int = 1,
        dilation: int = 1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.kernel_size = (3, 3)
        self.padding = (1, 1)
        self.stride = (stride, stride)
        self.dilation = (dilation, dilation)
        self.groups = in_channels
        self.padding_mode = 'zeros'

        gaussian = torch.tensor([[[
            [0.0811, 0.1226, 0.0811],
            [0.1226, 0.1853, 0.1226],
            [0.0811, 0.1226, 0.0811]
        ]], [[
            [-0.0811, -0.1226, -0.0811],
            [-0.1226, -0.1853, -0.1226],
            [-0.0811, -0.1226, -0.0811]
        ]]])

        self.weight = nn.Parameter(gaussian.repeat(
            self.in_channels // 2, 1, 1, 1), False)
        self.register_parameter('bias', None)

        self.weight.requires_grad_(False)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)


class InvertedResidualBlock(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        t,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        se_ratio: float = None,
        se_ind: bool = False,
        survival_prob: float = None,
        normalizer_fn: nn.Module = None,
        activation_fn: nn.Module = None,
        dw_se_act: nn.Module = None
    ):
        super().__init__()

        self.inp = inp
        self.planes = int(self.inp * t)
        self.oup = oup
        self.stride = stride
        self.apply_residual = (self.stride == 1) and (self.inp == self.oup)
        self.se_ratio = se_ratio if se_ind or se_ratio is None else (
            se_ratio / t)
        self.has_se = (self.se_ratio is not None) and (
            self.se_ratio > 0) and (self.se_ratio <= 1)

        normalizer_fn = normalizer_fn or _NORMALIZER
        activation_fn = activation_fn or _NONLINEAR

        layers = []
        if t != 1:
            layers.append(Conv2d1x1Block(
                inp, self.planes, normalizer_fn=normalizer_fn, activation_fn=activation_fn))

        if dw_se_act is None:
            layers.append(DepthwiseBlock(self.planes, self.planes, kernel_size, stride=self.stride,
                          padding=padding, dilation=dilation, normalizer_fn=normalizer_fn, activation_fn=activation_fn))
        else:
            layers.append(DepthwiseConv2dBN(self.planes, self.planes, kernel_size, stride=self.stride, padding=padding,
                                            dilation=dilation, normalizer_fn=normalizer_fn))

        if self.has_se:
            layers.append(SEBlock(self.planes, self.se_ratio))

        if dw_se_act:
            layers.append(dw_se_act())

        layers.append(
            Conv2d1x1BN(self.planes, oup, normalizer_fn=normalizer_fn)
        )

        if self.apply_residual and survival_prob:
            layers.append(DropPath(survival_prob))

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
        normalizer_fn: nn.Module = None,
        activation_fn: nn.Module = None
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

        normalizer_fn = normalizer_fn or _NORMALIZER
        activation_fn = activation_fn or _NONLINEAR

        layers = [
            Conv2dBlock(inp, self.planes, kernel_size, stride=self.stride, padding=self.padding,
                        normalizer_fn=normalizer_fn, activation_fn=activation_fn)
        ]

        if self.has_se:
            layers.append(SEBlock(self.planes, self.se_ratio))

        layers.append(Conv2d1x1BN(
            self.planes, oup, normalizer_fn=normalizer_fn))

        if self.apply_residual and survival_prob:
            layers.append(DropPath(survival_prob))

        self.branch1 = nn.Sequential(*layers)
        self.branch2 = nn.Identity() if self.apply_residual else None
        self.combine = Combine('ADD') if self.apply_residual else None

    def forward(self, x):
        if self.apply_residual:
            return self.combine([self.branch2(x), self.branch1(x)])
        else:
            return self.branch1(x)


class SharedDepthwiseConv2d(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        t: int = 2,
        bias: bool = False
    ):
        super().__init__()

        self.channels = channels // t
        self.t = t

        self.mux = DepthwiseConv2d(
            self.channels, self.channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        x = torch.chunk(x, self.t, dim=1)
        x = [self.mux(xi) for xi in x]
        return torch.cat(x, dim=1)


class MlpBlock(nn.Sequential):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation_fn: nn.Module = nn.GELU,
        dropout_rate: float = 0.
    ):
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        layers = OrderedDict([
            ('fc1', nn.Linear(in_features, hidden_features)),
            ('act', activation_fn()),
        ])

        if dropout_rate != 0.:
            layers['do1'] = nn.Dropout(dropout_rate)

        layers['fc2'] = nn.Linear(hidden_features, out_features)

        if dropout_rate != 0.:
            layers['do2'] = nn.Dropout(dropout_rate)

        super().__init__(layers)


class MultiHeadDotProductAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias: bool = False,
        attn_dropout_rate: float = 0.,
        proj_dropout_rate: float = 0.
    ):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.w_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.drop = nn.Dropout(attn_dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout_rate)

    def forward(self, x):
        # 1. The first step is to calculate the Query, Key, and Value matrices.
        #    We do that by packing our embeddings into a matrix X, and multiplying it by the weight matrices we’ve trained(WQ, WK, WV)
        B, N, C = x.shape
        qkv = self.w_qkv(x).reshape(B, N, 3, self.num_heads,
                                    C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B heads N head_dim

        # self-attention
        score = torch.einsum('bnqd, bnkd -> bnqk', q, k) * self.scale
        score = score.softmax(-1)
        score = self.drop(score)
        out = torch.einsum('bnsd, bndv -> bnsv', score, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)  # concat

        return self.proj_drop(self.proj(out))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads: int = 8,
        qkv_bias: bool = False,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.,
        attn_dropout_rate: float = 0.,
        drop_path_rate: float = 0.,
        normalizer_fn: nn.Module = nn.LayerNorm,
    ):
        super().__init__()

        self.msa = nn.Sequential(
            normalizer_fn(embed_dim),
            MultiHeadDotProductAttention(
                embed_dim, num_heads, qkv_bias,
                attn_dropout_rate=attn_dropout_rate, proj_dropout_rate=dropout_rate),
            DropPath(1 - drop_path_rate)
        )

        self.mlp = nn.Sequential(
            normalizer_fn(embed_dim),
            MlpBlock(embed_dim, int(embed_dim * mlp_ratio),
                     dropout_rate=dropout_rate),
            DropPath(1 - drop_path_rate)
        )

    def forward(self, x):
        x = x + self.msa(x)
        x = x + self.mlp(x)
        return x


class Filters32(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stride: int = 1,
        dilation: int = 1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.kernel_size = (3, 3)
        self.padding = (1, 1)
        self.stride = (stride, stride)
        self.dilation = (dilation, dilation)
        self.groups = in_channels
        self.padding_mode = 'zeros'

        sharpness = torch.tensor(
            [[[
                [-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]
            ]], [[
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ]], [[
                [-1, 0, -1],
                [0, 5, 0],
                [-1, 0, -1]
            ]]], dtype=torch.float32
        )

        edge = torch.tensor(
            [[[
                [-1, -1, -1],
                [0,  0,  0],
                [1,  1,  1]
            ]], [[
                [-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1]
            ]], [[
                [-1/2, -2/2, -1/2],
                [0, 0, 0],
                [1/2, 2/2, 1/2]
            ]], [[
                [-1/2, 0, 1/2],
                [-2/2, 0, 2/2],
                [-1/2, 0, 1/2]
            ]], [[
                [-1/2, 0, 0],
                [0, 2/2, 0],
                [0, 0, -1/2]
            ]], [[
                [0, 0, -1/2],
                [0, 2/2, 0],
                [-1/2, 0, 0]
            ]], [[
                [0, -1/2, 0],
                [0,  2/2, 0],
                [0, -1/2, 0]
            ]], [[
                [0, 0, 0],
                [-1/2,  2/2, -1/2],
                [0, 0, 0]
            ]], [[
                [0, -1/2, 0],
                [0, -1/2, 0],
                [0, 2/2, 0]
            ]], [[
                [0, 2/2, 0],
                [0, -1/2, 0],
                [0, -1/2, 0]
            ]], [[
                [0, 0, 0],
                [-1/2, -1/2, 2/2],
                [0, 0, 0]
            ]], [[
                [0, 0, 0],
                [2/2, -1/2, -1/2],
                [0, 0, 0]
            ]], [[
                [-1/8, -1/8, -1/8],
                [-1/8, 8/8, -1/8],
                [-1/8, -1/8, -1/8]
            ]], [[
                [-1, 0, 1],
                [0,  0,  0],
                [1,  0,  -1]
            ]]], dtype=torch.float32
        )

        embossing = torch.tensor(
            [[[
                [-1, -1, 0],
                [-1, 0, 1],
                [0, 1, 1]
            ]], [[
                [0, 1, 1],
                [-1, 0, 1],
                [-1, -1, 0]
            ]], [[
                [-3/3, -2/3, -1/3],
                [-2/3, 0, 2/3],
                [1/3, 2/3, 3/3]
            ]], [[
                [-1/3, -2/3, -3/3],
                [2/3, 0, -2/3],
                [3/3, 2/3, 1/3]
            ]], [[
                [0, -1, 0],
                [-1, 0, 1],
                [0, 1, 0]
            ]], [[
                [0, -1, 0],
                [1, 0, -1],
                [0, 1, 0]
            ]]], dtype=torch.float32
        )

        box = torch.tensor(
            [[[
                [1/9, 1/9, 1/9],
                [1/9, 1/9, 1/9],
                [1/9, 1/9, 1/9]
            ]], [[
                [.0, 1/5, .0],
                [1/5, 1/5, 1/5],
                [.0, 1/5, .0]
            ]], [[
                [1/5, 0, 1/5],
                [0, 1/5, 0],
                [1/5, 0, 1/5]
            ]]], dtype=torch.float32
        )

        gaussian = torch.tensor([[[
            [0.0811, 0.1226, 0.0811],
            [0.1226, 0.1853, 0.1226],
            [0.0811, 0.1226, 0.0811]
        ]], [[
            [0.0571, 0.1248, 0.0571],
            [0.1248, 0.2725, 0.1248],
            [0.0571, 0.1248, 0.0571]
        ]], [[
            [0.0439, 0.1217, 0.0439],
            [0.1217, 0.3377, 0.1217],
            [0.0439, 0.1217, 0.0439]
        ]], [[
            [0.0277, 0.1110, 0.0277],
            [0.1110, 0.4452, 0.1110],
            [0.0277, 0.1110, 0.0277]
        ]]], dtype=torch.float32)

        motion = torch.tensor(
            [[[
                [1/3, 0, 0],
                [0, 1/3, 0],
                [0, 0, 1/3]
            ]], [[
                [0, 0, 1/3],
                [0, 1/3, 0],
                [1/3, 0, 0]
            ]]], dtype=torch.float32
        )

        kernels = torch.cat(
            [sharpness, edge, embossing, box, gaussian, motion], dim=0)

        self.weight = nn.Parameter(kernels.repeat(
            self.in_channels // 32, 1, 1, 1), False)
        self.register_parameter('bias', None)

        self.weight.requires_grad_(False)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)
