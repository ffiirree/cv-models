import os
import torch
import torch.nn as nn
from .core import blocks
from typing import Any, List

__all__ = ['ResNet', 'resnet18', 'resnet34',
           'resnet50', 'resnet101', 'resnet152',
           'resnext50_32x4d', 'resnext101_32x8d']


def resnet18(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = ResNet(layers=[2, 2, 2, 2], block=blocks.ResBasicBlock, **kwargs)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def resnet34(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = ResNet(layers=[3, 4, 6, 3], block=blocks.ResBasicBlock, **kwargs)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def resnet50(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = ResNet(layers=[3, 4, 6, 3], block=blocks.Bottleneck, **kwargs)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def resnet101(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = ResNet(layers=[3, 4, 23, 3], block=blocks.Bottleneck, **kwargs)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def resnet152(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = ResNet(layers=[3, 8, 36, 3], block=blocks.Bottleneck, **kwargs)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def resnext50_32x4d(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = ResNet(
        layers=[3, 4, 6, 3],
        block=blocks.Bottleneck,
        groups=32,
        width_per_group=4,
        **kwargs
    )

    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))

    return model


def resnext101_32x8d(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = ResNet(
        layers=[3, 4, 23, 3],
        block=blocks.Bottleneck,
        groups=32,
        width_per_group=8,
        **kwargs
    )

    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))

    return model


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        layers: List[int] = [2, 2, 2, 2],
        groups: int = 1,
        width_per_group: int = 64,
        block: nn.Module = blocks.ResBasicBlock
    ):
        super().__init__()

        self.groups = groups
        self.width_per_group = width_per_group
        self.block = block

        features = [
            blocks.Conv2dBlock(in_channels, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]

        features.extend(self.make_layers(
            64 // block.expansion, 64, 1, layers[0]))
        features.extend(self.make_layers(64, 128, 2, layers[1]))
        features.extend(self.make_layers(128, 256, 2, layers[2]))
        features.extend(self.make_layers(256, 512, 2, layers[3]))

        self.features = nn.Sequential(*features)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def make_layers(self, inp, oup, stride, n):
        layers = [
            self.block(inp * self.block.expansion, oup, stride=stride,
                       groups=self.groups, width_per_group=self.width_per_group)
        ]
        for _ in range(n - 1):
            layers.append(self.block(oup * self.block.expansion, oup,
                          groups=self.groups, width_per_group=self.width_per_group))
        return layers
