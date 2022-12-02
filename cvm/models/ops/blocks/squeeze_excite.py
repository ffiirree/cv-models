from functools import partial
from contextlib import contextmanager
from torch import nn
from . import norm_act
from .vanilla_conv2d import Conv2d1x1
from ..functional import make_divisible
from typing import OrderedDict

_SE_INNER_NONLINEAR: nn.Module = partial(nn.ReLU, inplace=True)
_SE_GATING_FN: nn.Module = nn.Sigmoid
_SE_DIVISOR: int = 8
_SE_USE_NORM: bool = False


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


class SEBlock(nn.Sequential):
    """Squeeze-and-Excitation Block
    """

    def __init__(
        self,
        channels,
        ratio,
        inner_activation_fn: nn.Module = None,
        gating_fn: nn.Module = None
    ):
        squeezed_channels = make_divisible(int(channels * ratio), _SE_DIVISOR)
        inner_activation_fn = inner_activation_fn or _SE_INNER_NONLINEAR
        gating_fn = gating_fn or _SE_GATING_FN

        layers = OrderedDict([])

        layers['pool'] = nn.AdaptiveAvgPool2d((1, 1))
        layers['reduce'] = Conv2d1x1(channels, squeezed_channels, bias=True)
        if _SE_USE_NORM:
            layers['norm'] = norm_act.normalizer_fn(squeezed_channels)
        layers['act'] = inner_activation_fn()
        layers['expand'] = Conv2d1x1(squeezed_channels, channels, bias=True)
        layers['gate'] = gating_fn()

        super().__init__(layers)

    def _forward(self, input):
        for module in self:
            input = module(input)
        return input

    def forward(self, x):
        return x * self._forward(x)
