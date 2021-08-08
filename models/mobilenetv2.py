import os
import torch
import torch.nn as nn
from .core import blocks

__all__ = ['MobileNetV2', 'mobilenet_v2']


def mobilenet_v2(pretrained: bool = False, pth: str = None):
    model = MobileNetV2()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetV2(nn.Module):
    @blocks.nonlinear(nn.ReLU6)
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000
    ):
        super().__init__()

        t = [1, 6, 6, 6, 6, 6, 6]
        c = [32, 16, 24, 32, 64, 96, 160, 320, 1280]
        n = [1, 2, 3, 4, 3, 3, 1]
        s = [1, 2, 2, 2, 1, 2, 1]

        features = [blocks.Conv2dBlock(in_channels, c[0], 3, stride=2)]

        for i in range(len(t)):
            features.append(self.make_layers(c[i], t[i], c[i+1], n[i], s[i]))

        features.append(blocks.Conv2d1x1Block(c[-2], c[-1]))

        self.features = nn.Sequential(*features)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(c[-1], num_classes)
        )

    @staticmethod
    def make_layers(inp: int, t: int, oup: int, n: int, stride: int):
        layers = [
            blocks.InvertedResidualBlock(inp, oup, t, stride=stride)
        ]

        for _ in range(n - 1):
            layers.append(blocks.InvertedResidualBlock(oup, oup, t))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
