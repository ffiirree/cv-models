import torch
from torch import nn
from .vanilla_conv2d import Conv2d1x1
from .factory import normalizer_fn, activation_fn
from ..functional import make_divisible


class ChannelAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        rd_ratio: float = 1/8,
        rd_divisor: int = 8,
        gate_fn: nn.Module = nn.Sigmoid
    ) -> None:
        super().__init__()

        rd_channels = make_divisible(in_channels * rd_ratio, rd_divisor)

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp = nn.Sequential(
            Conv2d1x1(in_channels, rd_channels, bias=True),
            activation_fn(),
            Conv2d1x1(rd_channels, in_channels, bias=True)
        )
        self.gate = gate_fn()

    def forward(self, x):
        return x * self.gate(self.mlp(self.max_pool(x)) + self.mlp(self.avg_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(
        self,
        kernel_size: int = 7,
        gate_fn: nn.Module = nn.Sigmoid
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.norm = normalizer_fn(1)
        self.gate = gate_fn()

    def forward(self, x):
        s = torch.cat([torch.amax(x, dim=1, keepdim=True), torch.mean(x, dim=1, keepdim=True)], dim=1)
        return x * self.gate(self.norm(self.conv(s)))


class CBAM(nn.Sequential):
    r"""
    Paper: CBAM: Convolutional Block Attention Module, https://arxiv.org/abs/1807.06521
    Code: https://github.com/Jongchan/attention-module
    """

    def __init__(
        self,
        in_channels,
        rd_ratio,
        kernel_size: int = 7,
        gate_fn: nn.Module = nn.Sigmoid
    ) -> None:
        super().__init__(
            ChannelAttention(in_channels, rd_ratio, gate_fn=gate_fn),
            SpatialAttention(kernel_size=kernel_size, gate_fn=gate_fn)
        )
