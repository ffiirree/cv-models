import torch
import torch.nn as nn

from .ops import blocks
from .utils import export, config, load_from_local_or_url
from typing import Any


# Paper suggests 0.99 momentum
_BN_MOMENTUM = 0.01


@export
class MnasNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        dropout_rate: float = 0.2,
        thumbnail: bool = False,
        **kwargs: Any
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 2

        t = [1, 6, 3, 6, 6, 6, 6]
        c = [32, 16, 24, 40, 80, 112, 160, 320, 1280]
        n = [1, 2, 3, 4, 2, 3, 1]  # repeats
        s = [1, FRONT_S, 2, 2, 1, 2, 1]
        k = [3, 3, 5, 3, 3, 5, 3]
        se = [0, 0, 0.25, 0, 0.25, 0.25, 0]

        features = [blocks.Conv2dBlock(in_channels, c[0], 3, stride=FRONT_S)]

        for i in range(len(t)):
            features.append(
                self.make_layers(c[i], t[i], c[i+1], n[i], s[i], k[i], se[i])
            )

        features.append(blocks.Conv2d1x1Block(c[-2], c[-1]))

        self.features = nn.Sequential(*features)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate, inplace=True),
            nn.Linear(c[-1], num_classes)
        )

    @staticmethod
    def make_layers(
        inp: int,
        t: int,
        oup: int,
        n: int,
        stride: int,
        kernel_size: int = 3,
        rd_ratio: float = None
    ):
        layers = [blocks.InvertedResidualBlock(inp, oup, t, kernel_size, stride, rd_ratio=rd_ratio)]

        for _ in range(n - 1):
            layers.append(blocks.InvertedResidualBlock(oup, oup, t, kernel_size, rd_ratio=rd_ratio))

        return blocks.Stage(layers)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


@export
def mnasnet_a1(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    model = MnasNet(**kwargs)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model
