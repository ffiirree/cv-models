import os
import torch
import torch.nn as nn
from .core import blocks
from typing import Any, List

__all__ = ['VGGNet', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
           'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias):
        super().__init__(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, bias=bias),
            nn.ReLU(inplace=True)
        )


def vgg11(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = VGGNet(layers=[1, 1, 2, 2, 2], **kwargs)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def vgg13(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = VGGNet(layers=[2, 2, 2, 2, 2], **kwargs)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def vgg16(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = VGGNet(layers=[2, 2, 3, 3, 3], **kwargs)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def vgg19(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = VGGNet(layers=[2, 2, 4, 4, 4], **kwargs)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def vgg11_bn(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = VGGNet(layers=[1, 1, 2, 2, 2], block=blocks.Conv2dBlock, **kwargs)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def vgg13_bn(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = VGGNet(layers=[2, 2, 2, 2, 2], block=blocks.Conv2dBlock, **kwargs)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def vgg16_bn(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = VGGNet(layers=[2, 2, 3, 3, 3], block=blocks.Conv2dBlock, **kwargs)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


def vgg19_bn(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = VGGNet(layers=[2, 2, 4, 4, 4], block=blocks.Conv2dBlock, **kwargs)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class VGGNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        layers: List[int] = [1, 1, 2, 2, 2],
        block: nn.Module = Conv2dReLU,
        thumbnail: bool = False
    ):
        super().__init__()

        maxpool1 = nn.Identity() if thumbnail else nn.MaxPool2d(2, stride=2)
        maxpool2 = nn.Identity() if thumbnail else nn.MaxPool2d(2, stride=2)

        self.features = nn.Sequential(
            *self.make_layers(in_channels, 64, layers[0], block),
            maxpool1,
            *self.make_layers(64, 128, layers[1], block),
            maxpool2,
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
