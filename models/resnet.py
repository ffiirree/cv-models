import os
import torch
import torch.nn as nn
from .core import *

__all__ = ['ResNet', 'resnet18', 'resnet34',
           'resnet50', 'resnet101', 'resnet152']


def resnet18(pretrained: bool = False, pth: str = None):
    model = ResNet(layers=[2, 2, 2, 2], block=blocks.ResBasicBlock)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def resnet34(pretrained: bool = False, pth: str = None):
    model = ResNet(layers=[3, 4, 6, 3], block=blocks.ResBasicBlock)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def resnet50(pretrained: bool = False, pth: str = None):
    model = ResNet(layers=[3, 4, 6, 3], block=blocks.Bottleneck)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def resnet101(pretrained: bool = False, pth: str = None):
    model = ResNet(layers=[3, 4, 23, 3], block=blocks.Bottleneck)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def resnet152(pretrained: bool = False, pth: str = None):
    model = ResNet(layers=[3, 8, 36, 3], block=blocks.Bottleneck)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        layers: list = [2, 2, 2, 2],
        block: nn.Module = blocks.ResBasicBlock
    ):
        super().__init__()

        features = [
            blocks.Conv2dBlock(in_channels, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]

        features.extend(self.make_layers(
            64 // block.expansion, 64, 1, layers[0], block))
        features.extend(self.make_layers(64, 128, 2, layers[1], block))
        features.extend(self.make_layers(128, 256, 2, layers[2], block))
        features.extend(self.make_layers(256, 512, 2, layers[3], block))

        self.features = nn.Sequential(*features)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    @staticmethod
    def make_layers(inp, oup, stride, n, block):
        layers = [block(inp * block.expansion, oup, stride=stride)]
        for _ in range(n - 1):
            layers.append(block(oup * block.expansion,  oup))
        return layers
