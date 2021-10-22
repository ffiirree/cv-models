import os
import torch
import torch.nn as nn
from .core import blocks, export
from typing import Any, List, Type, Union


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias):
        super().__init__(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, bias=bias),
            nn.ReLU(inplace=True)
        )


def _vgg(
    layers: List[int],
    block: Type[Union[Conv2dReLU, blocks.Conv2dBlock]] = Conv2dReLU,
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    model = VGGNet(layers=layers, block=block, **kwargs)

    if pretrained:
        if pth is not None:
            state_dict = torch.load(os.path.expanduser(pth))
        else:
            assert 'url' in kwargs and kwargs['url'] != '', 'Invalid URL.'
            state_dict = torch.hub.load_state_dict_from_url(
                kwargs['url'],
                progress=progress
            )
        model.load_state_dict(state_dict)
    return model


@export
def vgg11(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vgg([1, 1, 2, 2, 2], Conv2dReLU, pretrained, pth, progress, **kwargs)


@export
def vgg13(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vgg([2, 2, 2, 2, 2], Conv2dReLU, pretrained, pth, progress, **kwargs)


@export
def vgg16(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vgg([2, 2, 3, 3, 3], Conv2dReLU, pretrained, pth, progress, **kwargs)


@export
def vgg19(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vgg([2, 2, 4, 4, 4], Conv2dReLU, pretrained, pth, progress, **kwargs)


@export
def vgg11_bn(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vgg([1, 1, 2, 2, 2], blocks.Conv2dBlock, pretrained, pth, progress, **kwargs)


@export
def vgg13_bn(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vgg([2, 2, 2, 2, 2], blocks.Conv2dBlock, pretrained, pth, progress, **kwargs)


@export
def vgg16_bn(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vgg([2, 2, 3, 3, 3], blocks.Conv2dBlock, pretrained, pth, progress, **kwargs)


@export
def vgg19_bn(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vgg([2, 2, 4, 4, 4], blocks.Conv2dBlock, pretrained, pth, progress, **kwargs)


@export
class VGGNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        layers: List[int] = [1, 1, 2, 2, 2],
        block: nn.Module = Conv2dReLU,
        dropout_rate: float = 0.5,
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
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
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
