
import torch
import torch.nn as nn
from .depthwise_separable_conv2d import PointwiseConv2dBN


class SCAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        rd_ratio
    ) -> None:
        super().__init__()

        planes = int(in_channels * rd_ratio)

        self.reduce = nn.Sequential(
            nn.AvgPool2d((3, 3), (2, 2)),
            # blocks.DepthwiseBlock(in_channels, in_channels, 3, stride=2),
            PointwiseConv2dBN(in_channels, planes, bias=True)
        )

        self.down1x1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            PointwiseConv2dBN(planes, planes, bias=True)
        )

        self.expand = PointwiseConv2dBN(planes, in_channels, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if len(list(m.parameters())) > 1:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 0.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        N, C, W, H = x.shape

        reduce = self.reduce(x)

        down_1x1 = self.down1x1(reduce)

        spatial = torch.nn.functional.interpolate(
            torch.sum(reduce * torch.softmax(down_1x1, dim=1), dim=1, keepdims=True),
            x.shape[-2:]
        ).view(N, 1, -1)

        channel = self.expand(down_1x1).squeeze(-1)

        score = torch.sigmoid(torch.einsum('nco, nox -> ncx', channel, spatial).view(N, C, W, H))

        return x * score
