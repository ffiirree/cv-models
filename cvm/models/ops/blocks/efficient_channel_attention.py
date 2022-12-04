import math
import torch
from torch import nn


class EfficientChannelAttention(nn.Module):
    r"""
    Paper: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks, https://arxiv.org/abs/1910.03151
    """
    def __init__(
        self,
        in_channels,
        gamma=2,
        beta=2
    ) -> None:
        super().__init__()

        t = int(abs((math.log(in_channels, 2) + beta) / gamma))
        k = max(t if t % 2 else t + 1, 3)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        y = self.pool(x)
        y = self.conv(y.view(y.shape[0], 1, -1))
        y = y.view(y.shape[0], -1, 1, 1)
        y = self.gate(y)

        return x * y.expand_as(x)
