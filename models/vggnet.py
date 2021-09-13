import os
import torch
import torch.nn as nn
from .core import *

__all__ = ['VGGNet', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
           'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias):
        super().__init__(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, bias=bias),
            nn.ReLU(inplace=True)
        )


def vgg11(pretrained: bool = False, pth: str = None):
    model = VGGNet(layers=[1, 1, 2, 2, 2])
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def vgg13(pretrained: bool = False, pth: str = None):
    model = VGGNet(layers=[2, 2, 2, 2, 2])
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def vgg16(pretrained: bool = False, pth: str = None):
    model = VGGNet(layers=[2, 2, 3, 3, 3])
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def vgg19(pretrained: bool = False, pth: str = None):
    model = VGGNet(layers=[2, 2, 4, 4, 4])
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def vgg11_bn(pretrained: bool = False, pth: str = None):
    model = VGGNet(layers=[1, 1, 2, 2, 2], block=blocks.Conv2dBlock)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def vgg13_bn(pretrained: bool = False, pth: str = None):
    model = VGGNet(layers=[2, 2, 2, 2, 2], block=blocks.Conv2dBlock)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def vgg16_bn(pretrained: bool = False, pth: str = None):
    model = VGGNet(layers=[2, 2, 3, 3, 3], block=blocks.Conv2dBlock)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def vgg19_bn(pretrained: bool = False, pth: str = None):
    model = VGGNet(layers=[2, 2, 4, 4, 4], block=blocks.Conv2dBlock)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class VGGNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        layers: list = [1, 1, 2, 2, 2],
        block: nn.Module = Conv2dReLU
    ):
        super().__init__()

        self.features = nn.Sequential(
            *self.make_layers(in_channels, 64, layers[0], block),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.make_layers(64, 128, layers[1], block),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.make_layers(128, 256, layers[2], block),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.make_layers(256, 512, layers[3], block),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.make_layers(512, 512, layers[4], block),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avg = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    @staticmethod
    def make_layers(inp, oup, n, block):
        layers = [block(inp, oup, kernel_size=3, padding=1, bias=True)]

        for _ in range(n - 1):
            layers.append(block(oup, oup, kernel_size=3, padding=1, bias=True))

        return layers
