import torch
import torch.nn as nn
import torch.nn.functional as F
from .vanilla_conv2d import Conv2d1x1, Conv2d1x1Block, Conv2dBlock
from .channel import Combine
from typing import List


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            Conv2d1x1Block(in_channels, out_channels)
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        rates: List[int] = [6, 12, 18]
    ):
        super().__init__()

        ms = [Conv2d1x1Block(in_channels, out_channels)]
        for rate in rates:
            ms.append(Conv2dBlock(in_channels, out_channels, padding=rate, dilation=rate))

        ms.append(ASPPPooling(in_channels, out_channels))
        self.ms = nn.ModuleList(ms)

        self.combine = Combine('CONCAT')
        self.conv1x1 = Conv2d1x1(out_channels * len(self.ms), out_channels)

    def forward(self, x):
        aspp = []
        for module in self.ms:
            aspp.append(module(x))

        x = self.combine(aspp)
        x = self.conv1x1(x)
        return x
