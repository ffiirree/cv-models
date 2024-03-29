import torch
from torch import nn
import torch.nn.functional as F
from . import factory
from ..functional import get_gaussian_kernels2d
from typing import Tuple


class GaussianBlur(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        sigma_range: Tuple[float, float] = (1.0, 1.0),
        normalize: bool = True,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1
    ):
        super().__init__()

        padding = padding or ((kernel_size - 1) * (dilation - 1) + kernel_size) // 2

        self.channels = channels
        self.kernel_size = (kernel_size, kernel_size)
        self.padding = (padding, padding)
        self.stride = (stride, stride)
        self.dilation = (dilation, dilation)
        self.padding_mode = 'zeros'
        self.sigma_range = sigma_range
        self.normalize = normalize

        self.register_buffer(
            'weight',
            get_gaussian_kernels2d(
                kernel_size,
                torch.linspace(self.sigma_range[0], self.sigma_range[1], self.channels).view(-1, 1, 1, 1),
                self.normalize
            )
        )

    def forward(self, x):
        return F.conv2d(x, self.weight, None, self.stride, self.padding, self.dilation, self.channels)

    @property
    def out_channels(self):
        return self.channels

    def extra_repr(self):
        s = ('{channels}, kernel_size={kernel_size}'
             ', sigma_range={sigma_range}, normalize={normalize}, stride={stride}')
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
        sigma_range: Tuple[float, float] = (1.0, 1.0),
        normalize: bool = True,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        normalizer_fn: nn.Module = None
    ):
        normalizer_fn = normalizer_fn or factory._NORMALIZER

        super().__init__(
            GaussianBlur(channels, kernel_size, sigma_range, normalize,
                         stride=stride, padding=padding, dilation=dilation),
            normalizer_fn(channels)
        )


class GaussianBlurBlock(nn.Sequential):
    def __init__(
        self,
        channels,
        kernel_size: int = 3,
        sigma_range: Tuple[float, float] = (1.0, 1.0),
        normalize: bool = True,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        normalizer_fn: nn.Module = None,
        activation_fn: nn.Module = None,
        norm_position: str = None
    ):
        super().__init__(
            GaussianBlur(channels, kernel_size, sigma_range, normalize,
                         stride=stride, padding=padding, dilation=dilation),
            *factory.norm_activation(channels, normalizer_fn, activation_fn, norm_position)
        )
