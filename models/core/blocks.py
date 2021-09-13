from typing import List, OrderedDict
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
from .functional import *

_BN_MOMENTUM: float = 0.1
_BN_EPSILON: float = 1e-5
_BN_POSIITON: str = 'before'
_NONLINEAR: nn.Module = nn.ReLU
_SE_GATING_FN: nn.Module = nn.Sigmoid


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


@contextmanager
def se_gating_fn(fn: nn.Module):
    global _SE_GATING_FN

    _pre_fn = _SE_GATING_FN
    _SE_GATING_FN = fn
    yield
    _SE_GATING_FN = _pre_fn


def norm_activation(
    channels,
    bn_epsilon: float = None,
    bn_momentum: float = None,
    normalizer_fn: nn.Module = nn.BatchNorm2d,
    activation_fn: nn.Module = None,
    norm_position: str = None
) -> List[nn.Module]:
    norm_position = norm_position if norm_position else _BN_POSIITON
    assert norm_position in ['before', 'after', 'none'], ''

    bn_epsilon = bn_epsilon if bn_epsilon else _BN_EPSILON
    bn_momentum = bn_momentum if bn_momentum else _BN_MOMENTUM
    activation_fn = activation_fn if activation_fn else _NONLINEAR

    layers = [
        normalizer_fn(channels, eps=bn_epsilon, momentum=bn_momentum),
        activation_fn(inplace=True)
    ]

    if norm_position == 'before':
        return layers
    elif norm_position == 'after':
        layers.reverse()
        return layers

    return [activation_fn(inplace=True)]


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
        normalizer_fn: nn.Module = nn.BatchNorm2d
    ):
        bn_epsilon = bn_epsilon if bn_epsilon else _BN_EPSILON
        bn_momentum = bn_momentum if bn_momentum else _BN_MOMENTUM

        super().__init__(
            Conv2d3x3(in_channels, out_channels, stride=stride,
                      padding=padding, bias=bias, groups=groups),
            normalizer_fn(out_channels, eps=bn_epsilon, momentum=bn_momentum)
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
        normalizer_fn: nn.Module = nn.BatchNorm2d
    ):
        bn_epsilon = bn_epsilon if bn_epsilon else _BN_EPSILON
        bn_momentum = bn_momentum if bn_momentum else _BN_MOMENTUM

        super().__init__(
            Conv2d1x1(in_channels, out_channels, stride=stride,
                      padding=padding, bias=bias, groups=groups),
            normalizer_fn(out_channels, eps=bn_epsilon, momentum=bn_momentum)
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
        normalizer_fn: nn.Module = nn.BatchNorm2d,
        activation_fn: nn.Module = None,
        norm_position: str = None
    ):
        super().__init__(
            Conv2d1x1(in_channels, out_channels, stride=stride,
                      padding=padding, bias=bias, groups=groups),
            *norm_activation(out_channels, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum,
                             normalizer_fn=normalizer_fn, activation_fn=activation_fn,
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
        bias: bool = False,
        groups: int = 1,
        bn_epsilon: float = None,
        bn_momentum: float = None,
        normalizer_fn: nn.Module = nn.BatchNorm2d,
        activation_fn: nn.Module = None,
        norm_position: str = None,
    ):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      bias=bias, stride=stride, padding=padding, groups=groups),
            *norm_activation(out_channels, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum,
                             normalizer_fn=normalizer_fn, activation_fn=activation_fn,
                             norm_position=norm_position)
        )


class ResBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inp,
        oup,
        stride: int = 1,
        groups: int = 1,
        width_per_group: int = 64,
        bn_epsilon: float = None,
        bn_momentum: float = None,
        normalizer_fn: nn.Module = nn.BatchNorm2d,
        activation_fn: nn.Module = None
    ):
        super().__init__()

        bn_epsilon = bn_epsilon if bn_epsilon else _BN_EPSILON
        bn_momentum = bn_momentum if bn_momentum else _BN_MOMENTUM
        activation_fn = activation_fn if activation_fn else _NONLINEAR

        if width_per_group != 64:
            raise ValueError('width_per_group are not supported!')

        self.branch1 = nn.Sequential(OrderedDict([
            ('conv1', Conv2d3x3(inp, oup, stride=stride, groups=groups)),
            ('norm1', normalizer_fn(oup, eps=bn_epsilon, momentum=bn_momentum)),
            ('relu1', activation_fn(inplace=True)),
            ('conv2', Conv2d3x3(oup, oup)),
            ('norm2', normalizer_fn(oup, eps=bn_epsilon, momentum=bn_momentum))
        ]))

        self.branch2 = nn.Identity()

        if inp != oup or stride != 1:
            self.branch2 = Conv2d1x1BN(
                inp, oup, stride=stride, bn_momentum=bn_momentum, normalizer_fn=normalizer_fn)

        self.combine = Combine('ADD')
        self.relu = activation_fn(inplace=True)

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
        groups: int = 1,
        width_per_group: int = 64,
        bn_epsilon: float = None,
        bn_momentum: float = None,
        normalizer_fn: nn.Module = nn.BatchNorm2d,
        activation_fn: nn.Module = None
    ):
        super().__init__()

        bn_epsilon = bn_epsilon if bn_epsilon else _BN_EPSILON
        bn_momentum = bn_momentum if bn_momentum else _BN_MOMENTUM
        activation_fn = activation_fn if activation_fn else _NONLINEAR

        width = int(oup * (width_per_group / 64)) * groups

        self.branch1 = nn.Sequential(OrderedDict([
            ('conv1', Conv2d1x1(inp, width)),
            ('norm1', normalizer_fn(width, eps=bn_epsilon, momentum=bn_momentum)),
            ('relu1', activation_fn(inplace=True)),
            ('conv2', Conv2d3x3(width, width, stride=stride, groups=groups)),
            ('norm2', normalizer_fn(width, eps=bn_epsilon, momentum=bn_momentum)),
            ('relu2', activation_fn(inplace=True)),
            ('conv3', Conv2d1x1(width, oup * self.expansion)),
            ('norm3', normalizer_fn(oup * self.expansion, eps=bn_epsilon, momentum=bn_momentum)),
        ]))

        self.branch2 = nn.Identity()

        if stride != 1 or inp != oup * self.expansion:
            self.branch2 = Conv2d1x1BN(
                inp, oup * self.expansion, stride=stride, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, normalizer_fn=normalizer_fn)

        self.combine = Combine('ADD')
        self.relu = activation_fn(inplace=True)

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
        normalizer_fn: nn.Module = nn.BatchNorm2d
    ):
        bn_epsilon = bn_epsilon if bn_epsilon else _BN_EPSILON
        bn_momentum = bn_momentum if bn_momentum else _BN_MOMENTUM

        super().__init__(
            DepthwiseConv2d(inp, oup, kernel_size,
                            stride=stride, padding=padding),
            normalizer_fn(oup, eps=bn_epsilon, momentum=bn_momentum)
        )


class PointwiseConv2dBN(nn.Sequential):
    def __init__(
        self,
        inp,
        oup,
        stride: int = 1,
        bn_epsilon: float = None,
        bn_momentum: float = None,
        normalizer_fn: nn.Module = nn.BatchNorm2d
    ):
        bn_epsilon = bn_epsilon if bn_epsilon else _BN_EPSILON
        bn_momentum = bn_momentum if bn_momentum else _BN_MOMENTUM

        super().__init__(
            PointwiseConv2d(inp, oup, stride=stride),
            normalizer_fn(oup, eps=bn_epsilon, momentum=bn_momentum)
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
        normalizer_fn: nn.Module = nn.BatchNorm2d,
        activation_fn: nn.Module = None,
        norm_position: str = None
    ):
        super().__init__(
            DepthwiseConv2d(inp, oup, kernel_size,
                            stride=stride, padding=padding),
            *norm_activation(oup, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum,
                             normalizer_fn=normalizer_fn, activation_fn=activation_fn,
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
        normalizer_fn: nn.Module = nn.BatchNorm2d,
        activation_fn: nn.Module = None,
        norm_position: str = None,
    ):
        super().__init__(
            PointwiseConv2d(inp, oup, stride=stride, groups=groups),
            *norm_activation(oup, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum,
                             normalizer_fn=normalizer_fn, activation_fn=activation_fn,
                             norm_position=norm_position)
        )


class SEBlock(nn.Module):
    """Squeeze excite block
    """

    def __init__(
        self,
        channels,
        ratio,
        inner_activation_fn: nn.Module = nn.ReLU,
        gating_fn: nn.Module = None
    ):
        super().__init__()

        squeezed_channels = make_divisible(int(channels * ratio), 8)
        gating_fn = gating_fn if gating_fn else _SE_GATING_FN

        self.se = nn.Sequential(OrderedDict([
            ('pooling', nn.AdaptiveAvgPool2d((1, 1))),
            ('reduce', Conv2d1x1(channels, squeezed_channels, bias=True)),
            ('relu', inner_activation_fn(inplace=True)),
            ('expand', Conv2d1x1(squeezed_channels, channels, bias=True)),
            ('sigmoid', gating_fn()),
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


class OrthogonalBasisConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stride: int = 1,
        dilation: int = 1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = self.in_channels * 9
        self.kernel_size = (3, 3)
        self.padding = (1, 1)
        self.stride = (stride, stride)
        self.dilation = (dilation, dilation)
        self.groups = in_channels
        self.padding_mode = 'zeros'

        basis: torch.Tensor = torch.zeros([9, 1, *self.kernel_size])
        for i in range(self.kernel_size[0]):
            for j in range(self.kernel_size[1]):
                basis[i * self.kernel_size[0] + j, 0, i, j] = 1

        self.weight = nn.Parameter(basis.repeat(
            self.in_channels, 1, 1, 1), False)
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


class Shift(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        blocks = []

        blocks.append(torch.roll(x, (1, 1), (2, 3)))
        blocks.append(torch.roll(x, 1, 2))
        blocks.append(torch.roll(x, (1, -1), (2, 3)))
        blocks.append(torch.roll(x, 1, 3))
        blocks.append(x)
        blocks.append(torch.roll(x, -1, 3))
        blocks.append(torch.roll(x, (-1, 1), (2, 3)))
        blocks.append(torch.roll(x, -1, 2))
        blocks.append(torch.roll(x, (-1, -1), (2, 3)))
        return torch.cat(blocks, 1)


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
        normalizer_fn: nn.Module = nn.BatchNorm2d,
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
        bn_epsilon = bn_epsilon if bn_epsilon else _BN_EPSILON
        bn_momentum = bn_momentum if bn_momentum else _BN_MOMENTUM
        activation_fn = activation_fn if activation_fn else _NONLINEAR

        layers = []
        if t != 1:
            layers.append(Conv2d1x1Block(
                inp, self.planes,
                bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, normalizer_fn=normalizer_fn,
                activation_fn=activation_fn))

        layers.append(DepthwiseBlock(self.planes, self.planes, kernel_size, stride=self.stride, padding=self.padding,
                                     bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, normalizer_fn=normalizer_fn, activation_fn=activation_fn))

        if self.has_se:
            layers.append(SEBlock(self.planes, self.se_ratio))

        layers.append(Conv2d1x1BN(
            self.planes, oup, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, normalizer_fn=normalizer_fn))

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


class HalfDepthwiseBlock(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bn_epsilon: float = None,
        bn_momentum: float = None,
        normalizer_fn: nn.Module = nn.BatchNorm2d,
        activation_fn: nn.Module = None,
        norm_position: str = None
    ):
        super().__init__()
        self.split = ChannelChunk(2)
        self.branch1 = nn.Identity()

        self.branch2 = nn.Sequential(
            DepthwiseConv2d(inp // 2, oup // 2, kernel_size,
                            stride=stride, padding=padding),
            *norm_activation(oup // 2, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum,
                             normalizer_fn=normalizer_fn, activation_fn=activation_fn,
                             norm_position=norm_position)
        )
        self.concat = Combine('CONCAT')

    def forward(self, x):
        x1, x2 = self.split(x)
        out = self.concat([self.branch1(x1), self.branch2(x2)])
        return out


class HalfDWInvertedResidualBlock(nn.Module):
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
        normalizer_fn: nn.Module = nn.BatchNorm2d,
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
        bn_epsilon = bn_epsilon if bn_epsilon else _BN_EPSILON
        bn_momentum = bn_momentum if bn_momentum else _BN_MOMENTUM
        activation_fn = activation_fn if activation_fn else _NONLINEAR

        layers = []
        if t != 1:
            layers.append(Conv2d1x1Block(
                inp, self.planes,
                bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, normalizer_fn=normalizer_fn,
                activation_fn=activation_fn))

        if stride == 1:
            layers.append(HalfDepthwiseBlock(self.planes, self.planes, kernel_size, stride=self.stride, padding=self.padding,
                                             bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, normalizer_fn=normalizer_fn, activation_fn=activation_fn))
        else:
            layers.append(DepthwiseBlock(self.planes, self.planes, kernel_size, stride=self.stride, padding=self.padding,
                                         bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, normalizer_fn=normalizer_fn, activation_fn=activation_fn))

        if self.has_se:
            layers.append(SEBlock(self.planes, self.se_ratio))

        layers.append(Conv2d1x1BN(
            self.planes, oup, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, normalizer_fn=normalizer_fn))

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


class IdentityDepthwiseBlock(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bn_epsilon: float = None,
        bn_momentum: float = None,
        normalizer_fn: nn.Module = nn.BatchNorm2d,
        activation_fn: nn.Module = None,
        norm_position: str = None
    ):
        super().__init__()
        self.branch1 = nn.Identity()

        self.branch2 = nn.Sequential(
            DepthwiseConv2d(inp, oup, kernel_size,
                            stride=stride, padding=padding),
            *norm_activation(oup, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum,
                             normalizer_fn=normalizer_fn, activation_fn=activation_fn,
                             norm_position=norm_position)
        )
        self.concat = Combine('CONCAT')

    def forward(self, x):
        out = self.concat([self.branch1(x), self.branch2(x)])
        return out


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
        normalizer_fn: nn.Module = nn.BatchNorm2d,
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
        bn_epsilon = bn_epsilon if bn_epsilon else _BN_EPSILON
        bn_momentum = bn_momentum if bn_momentum else _BN_MOMENTUM
        activation_fn = activation_fn if activation_fn else _NONLINEAR

        layers = [
            Conv2dBlock(inp, self.planes, kernel_size, stride=self.stride, padding=self.padding,
                        bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, normalizer_fn=normalizer_fn, activation_fn=activation_fn)
        ]

        if self.has_se:
            layers.append(SEBlock(self.planes, self.se_ratio))

        layers.append(Conv2d1x1BN(
            self.planes, oup, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, normalizer_fn=normalizer_fn))

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