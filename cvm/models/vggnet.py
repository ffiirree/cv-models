import torch
import torch.nn as nn
from .core import blocks, export, load_from_local_or_url
from typing import Any, List


@export
class VGGNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        layers: List[int] = [1, 1, 2, 2, 2],
        dropout_rate: float = 0.5,
        thumbnail: bool = False,
        **kwargs: Any
    ):
        super().__init__()

        maxpool1 = nn.Identity() if thumbnail else nn.MaxPool2d(2, stride=2)
        maxpool2 = nn.Identity() if thumbnail else nn.MaxPool2d(2, stride=2)

        self.features = nn.Sequential(
            *self.make_layers(in_channels, 64, layers[0]),
            maxpool1,
            *self.make_layers(64, 128, layers[1]),
            maxpool2,
            *self.make_layers(128, 256, layers[2]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.make_layers(256, 512, layers[3]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.make_layers(512, 512, layers[4]),
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

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    @staticmethod
    def make_layers(inp, oup, n):
        layers = [blocks.Conv2dBlock(inp, oup, bias=True)]

        for _ in range(n - 1):
            layers.append(blocks.Conv2dBlock(oup, oup, bias=True))

        return layers


def _vgg(
    layers: List[int],
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    model = VGGNet(layers=layers, **kwargs)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
@blocks.normalizer(None)
def vgg11(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vgg([1, 1, 2, 2, 2], pretrained, pth, progress, **kwargs)


@export
@blocks.normalizer(None)
def vgg13(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vgg([2, 2, 2, 2, 2], pretrained, pth, progress, **kwargs)


@export
@blocks.normalizer(None)
def vgg16(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vgg([2, 2, 3, 3, 3], pretrained, pth, progress, **kwargs)


@export
@blocks.normalizer(None)
def vgg19(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vgg([2, 2, 4, 4, 4], pretrained, pth, progress, **kwargs)


@export
def vgg11_bn(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vgg([1, 1, 2, 2, 2], pretrained, pth, progress, **kwargs)


@export
def vgg13_bn(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vgg([2, 2, 2, 2, 2], pretrained, pth, progress, **kwargs)


@export
def vgg16_bn(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vgg([2, 2, 3, 3, 3], pretrained, pth, progress, **kwargs)


@export
def vgg19_bn(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vgg([2, 2, 4, 4, 4], pretrained, pth, progress, **kwargs)
