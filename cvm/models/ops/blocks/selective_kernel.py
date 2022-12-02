import torch
from torch import nn
from .vanilla_conv2d import Conv2d1x1, Conv2d1x1Block
from .depthwise_separable_conv2d import DepthwiseBlock
from .channel_combine import Combine


class SelectiveKernelBlock(nn.Module):
    r"""
    Paper: Selective Kernel Networks, https://arxiv.org/abs/1903.06586
    """

    def __init__(
        self,
        in_channels,
        rd_ratio
    ) -> None:
        super().__init__()

        self.in_channels = in_channels

        rd_channels = max(int(in_channels * rd_ratio), 32)

        self.conv3x3 = DepthwiseBlock(in_channels, in_channels, kernel_size=3, dilation=1)
        self.conv5x5 = DepthwiseBlock(in_channels, in_channels, kernel_size=3, dilation=2)

        self.fuse = Combine('ADD')

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.reduce = Conv2d1x1Block(in_channels, rd_channels)

        self.qk = Conv2d1x1(rd_channels, in_channels * 2, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        u3 = self.conv3x3(x)
        u5 = self.conv5x5(x)

        u = self.fuse([u3, u5])

        s = self.pool(u)

        z = self.reduce(s)

        ab = self.softmax(self.qk(z).view(-1, 2, self.in_channels, 1, 1))

        v = torch.sum(torch.stack([u3, u5], dim=1) * ab, dim=1)

        return v
