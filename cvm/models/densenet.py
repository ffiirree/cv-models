import torch
import torch.nn as nn
from .core import blocks, export, load_from_local_or_url
from typing import Any, OrderedDict, List


class DenseLayer(nn.Sequential):
    '''BN-ReLU-Conv'''

    def __init__(self, inp, oup):
        super().__init__()

        super().__init__(OrderedDict([
            ('norm1', nn.BatchNorm2d(inp)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv1', blocks.Conv2d1x1(inp, oup)),
            ('norm2', nn.BatchNorm2d(oup)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv2', blocks.Conv2d3x3(oup, 32))
        ]))


class TransitionLayer(nn.Sequential):
    '''BN-ReLU-Conv'''

    def __init__(self, inp, oup):
        super().__init__(OrderedDict([
            ('norm', nn.BatchNorm2d(inp)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv', blocks.Conv2d1x1(inp, oup)),
            ('pool', nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
        ]))


class DenseBlock(nn.Module):
    def __init__(self, inp, oup, n):
        super().__init__()

        layers = []

        for i in range(n):
            layers.append(DenseLayer(inp + 32 * i, oup))

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        outs = [x]
        for layer in self.features.children():
            outs.append(layer(torch.cat(outs, dim=1)))
        return torch.cat(outs, dim=1)


@export
class DenseNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        layers: List[int] = [2, 2, 2, 2],
        channels: List[int] = [64, 128, 256, 512],
        thumbnail: bool = False,
        **kwargs: Any
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 2

        maxpool = nn.Identity()
        if not thumbnail:
            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, channels[0], 7, FRONT_S, padding=3),
            maxpool,
            DenseBlock(channels[0], 128, layers[0]),
            TransitionLayer(channels[0] + 32 * layers[0], channels[1]),
            DenseBlock(channels[1], 128, layers[1]),
            TransitionLayer(channels[1] + 32 * layers[1], channels[2]),
            DenseBlock(channels[2], 128, layers[2]),
            TransitionLayer(channels[2] + 32 * layers[2], channels[3]),
            DenseBlock(channels[3], 128, layers[3]),

            nn.BatchNorm2d(channels[3] + 32 * layers[-1]),
            nn.ReLU(inplace=True)
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels[3] + 32 * layers[-1], num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _densenet(
    layers: List[int],
    channels: List[int],
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    model = DenseNet(layers=layers, channels=channels, **kwargs)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
def densenet121(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _densenet([6, 12, 24, 16], [64, 128, 256, 512], pretrained, pth, progress, **kwargs)


@export
def densenet169(pretrained: bool = False, pth: str = None, progress: bool = True,  **kwargs: Any):
    return _densenet([6, 12, 32, 32], [64, 128, 256, 640], pretrained, pth, progress, **kwargs)


@export
def densenet201(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _densenet([6, 12, 48, 32], [64, 128, 256, 896], pretrained, pth, progress, **kwargs)


@export
def densenet264(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _densenet([6, 12, 64, 48], [64, 128, 256, 1408], pretrained, pth, progress, **kwargs)
