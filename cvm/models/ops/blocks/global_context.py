import torch
from torch import nn
from .vanilla_conv2d import Conv2d1x1
from .norm import LayerNorm2d
from ..functional import make_divisible


class GlobalContextBlock(nn.Module):
    r"""
    Paper: GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond, https://arxiv.org/abs/1904.11492
    """

    def __init__(
        self,
        in_channels,
        rd_ratio: float = 1/8,
        rd_divisor: int = 8,
    ) -> None:
        super().__init__()

        channels = make_divisible(in_channels * rd_ratio, rd_divisor)

        self.conv1x1 = Conv2d1x1(in_channels, 1, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.transform = nn.Sequential(
            Conv2d1x1(in_channels, channels),
            LayerNorm2d(channels),
            nn.ReLU(inplace=True),
            Conv2d1x1(channels, in_channels)
        )

    def forward(self, x):
        # context modeling
        c = torch.einsum(
            "ncx, nxo -> nco",
            x.view(x.shape[0], x.shape[1], -1),
            self.softmax(self.conv1x1(x).view(x.shape[0], -1, 1))
        )
        c = x * c.unsqueeze(-1)

        # transform
        return x + self.transform(c)
