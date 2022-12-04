import math
from functools import partial
from contextlib import contextmanager
from torch import nn
from .depthwise_separable_conv2d import DepthwiseBlock, DepthwiseConv2dBN
from .interpolate import Interpolate

_GE_INNER_NONLINEAR: nn.Module = partial(nn.ReLU, inplace=True)
_GE_GATING_FN: nn.Module = nn.Sigmoid


@contextmanager
def ge(
    inner_nonlinear: nn.Module = _GE_INNER_NONLINEAR,
    gating_fn: nn.Module = _GE_GATING_FN
):
    global _GE_INNER_NONLINEAR
    global _GE_GATING_FN

    _pre_inner_fn = _GE_INNER_NONLINEAR
    _pre_fn = _GE_GATING_FN

    _GE_INNER_NONLINEAR = inner_nonlinear
    _GE_GATING_FN = gating_fn

    yield

    _GE_INNER_NONLINEAR = _pre_inner_fn
    _GE_GATING_FN = _pre_fn


class GatherExciteBlock(nn.Module):
    r"""Gather-Excite Block
    Paper: Gather-Excite: Exploiting Feature Context in Convolutional Neural Networks, https://arxiv.org/abs/1810.12348
    Code: https://github.com/hujie-frank/GENet
    """

    def __init__(
        self,
        channels,
        extent_ratio: int = 0,
        param_free: bool = True,
        kernel_size: int = 3,
        inner_activation_fn: nn.Module = None,
        gating_fn: nn.Module = None
    ):
        super().__init__()

        inner_activation_fn = inner_activation_fn or _GE_INNER_NONLINEAR
        gating_fn = gating_fn or _GE_GATING_FN

        self.gather = nn.Sequential()

        if param_free is True:
            if extent_ratio == 0:
                self.gather = nn.AdaptiveAvgPool2d((1, 1))
            else:
                self.gather = nn.AvgPool2d((15, 15), stride=extent_ratio)
        else:
            if extent_ratio == 0:
                self.gather.append(DepthwiseConv2dBN(channels, channels, kernel_size=kernel_size, padding=0))
            else:
                for i in range(int(math.log2(extent_ratio))):
                    if i != (int(math.log2(extent_ratio)) - 1):
                        self.gather.append(DepthwiseBlock(channels, channels, kernel_size=kernel_size,
                                                          stride=2, activation_fn=inner_activation_fn))
                    else:
                        self.gather.append(DepthwiseConv2dBN(channels, channels, kernel_size=kernel_size, stride=2))

        self.excite = Interpolate()
        self.gate = gating_fn()

    def _forward(self, x):
        size = x.shape[-2:]

        # gather
        x = self.gather(x)

        if x.shape[-1] != 1:
            x = self.excite(x, size)

        x = self.gate(x)

        return x

    def forward(self, x):
        return x * self._forward(x)
