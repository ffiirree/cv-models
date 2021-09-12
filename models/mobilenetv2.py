import os
import torch
import torch.nn as nn
from .core import blocks
from .core.functional import make_divisible

__all__ = ['MobileNetV2', 'mobilenet_v2_x1_0',
           'mobilenet_v2_x0_75', 'mobilenet_v2_x0_5', 'mobilenet_v2_x0_35',
           'mobilenet_v2_b', 'mobilenet_v2_c']


def mobilenet_v2_x1_0(pretrained: bool = False, pth: str = None):
    model = MobileNetV2()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def mobilenet_v2_x0_75(pretrained: bool = False, pth: str = None):
    model = MobileNetV2(multiplier=0.75)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def mobilenet_v2_x0_5(pretrained: bool = False, pth: str = None):
    model = MobileNetV2(multiplier=0.5)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def mobilenet_v2_x0_35(pretrained: bool = False, pth: str = None):
    model = MobileNetV2(multiplier=0.35)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetV2(nn.Module):
    @blocks.nonlinear(nn.ReLU6)
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        multiplier: float = 1.0
    ):
        super().__init__()

        t = [1, 6, 6, 6, 6, 6, 6]
        c = [32, 16, 24, 32, 64, 96, 160, 320]
        n = [1, 2, 3, 4, 3, 3, 1]
        s = [1, 2, 2, 2, 1, 2, 1]

        if multiplier < 1.0:
            c = [make_divisible(x * multiplier, 8) for x in c]

        features = [blocks.Conv2dBlock(in_channels, c[0], 3, stride=2)]

        for i in range(len(t)):
            features.append(self.make_layers(c[i], t[i], c[i+1], n[i], s[i]))

        features.append(blocks.Conv2d1x1Block(c[-1], 1280))

        self.features = nn.Sequential(*features)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(1280, num_classes)
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


def mobilenet_v2_b(pretrained: bool = False, pth: str = None):
    model = MobileNetV2B()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetV2B(nn.Module):
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
            blocks.HalfDWInvertedResidualBlock(inp, oup, t, stride=stride)
        ]

        for _ in range(n - 1):
            layers.append(blocks.HalfDWInvertedResidualBlock(oup, oup, t))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def mobilenet_v2_c(pretrained: bool = False, pth: str = None):
    model = MobileNetV2C()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetV2C(nn.Module):
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
            blocks.HalfInvertedResidualBlock(inp, oup, t, stride=stride)
        ]

        for _ in range(n - 1):
            layers.append(blocks.HalfInvertedResidualBlock(oup, oup, t))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
