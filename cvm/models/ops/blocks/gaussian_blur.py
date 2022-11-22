import torch
from torch import nn
import torch.nn.functional as F
from . import norm_act
from ..functional import get_3x3_gaussian_weight2d


class GaussianBlur(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        learnable: bool = True
    ):
        super().__init__()

        padding = padding or ((kernel_size - 1) * (dilation - 1) + kernel_size) // 2

        self.channels = channels
        self.kernel_size = (kernel_size, kernel_size)
        self.padding = (padding, padding)
        self.stride = (stride, stride)
        self.dilation = (dilation, dilation)
        self.padding_mode = 'zeros'
        self.learnable = learnable

        self.sigma = nn.Parameter(torch.ones(channels), self.learnable)

        self.standard_w = None if self.learnable else nn.Parameter(
            get_3x3_gaussian_weight2d(torch.ones(channels)), False)

    def forward(self, x):
        return F.conv2d(x, self.weight if self.learnable else self.standard_w, None, self.stride, self.padding, self.dilation, self.channels)

    @property
    def weight(self):
        return get_3x3_gaussian_weight2d(self.sigma)

    @property
    def out_channels(self):
        return self.channels

    def extra_repr(self):
        s = ('{channels}, kernel_size={kernel_size}'
             ', learnable={learnable}, stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)


class GaussianBlurBN(nn.Sequential):
    def __init__(
        self,
        channels,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        learnable: bool = True,
        normalizer_fn: nn.Module = None
    ):
        normalizer_fn = normalizer_fn or norm_act._NORMALIZER

        super().__init__(
            GaussianBlur(channels, kernel_size, stride, padding, dilation, learnable=learnable),
            normalizer_fn(channels)
        )


class GaussianBlurBlock(nn.Sequential):
    def __init__(
        self,
        channels,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        learnable: bool = True,
        normalizer_fn: nn.Module = None,
        activation_fn: nn.Module = None,
        norm_position: str = None
    ):
        super().__init__(
            GaussianBlur(channels, kernel_size, stride, padding, dilation, learnable=learnable),
            *norm_act.norm_activation(channels, normalizer_fn, activation_fn, norm_position)
        )
