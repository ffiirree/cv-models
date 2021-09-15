import os
import torch
import torch.nn as nn
from .core import blocks
from typing import OrderedDict, Any

__all__ = ['Xception', 'xception']


def xception(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = Xception(**kwargs)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class SeparableConv2d(nn.Sequential):
    def __init__(self, inplanes, planes):
        super().__init__(
            blocks.DepthwiseConv2d(inplanes, inplanes),
            blocks.PointwiseConv2d(inplanes, planes),
            nn.BatchNorm2d(planes)
        )


class XceptionBlock(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        stride: int = 1,
        expand_first: bool = True,
        first_relu: bool = True
    ):
        super().__init__()

        layers = OrderedDict([])
        if first_relu:
            layers['relu1'] = nn.ReLU(inplace=True)

        planes = oup if expand_first else inp

        layers['conv1'] = SeparableConv2d(inp, planes)
        layers['relu2'] = nn.ReLU(inplace=True)
        layers['conv2'] = SeparableConv2d(planes, oup)

        self.branch1 = nn.Sequential(layers)

        self.branch2 = nn.Identity()

        if stride != 1:
            self.branch1.add_module('maxpool', nn.MaxPool2d(3, 2, padding=1))
            self.branch2 = nn.Sequential(
                blocks.PointwiseConv2d(inp, oup, stride=2),
                nn.BatchNorm2d(oup)
            )
        else:
            self.branch1.add_module('relu3', nn.ReLU(inplace=True))
            self.branch1.add_module('conv3', SeparableConv2d(oup, oup))

        self.combine = blocks.Combine('ADD')

    def forward(self, x):
        return self.combine([self.branch1(x), self.branch2(x)])


class Xception(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        small_input: bool = False
    ):
        super().__init__()

        FRONT_S = 1 if small_input else 2

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, 32, stride=FRONT_S, padding=0),
            blocks.Conv2dBlock(32, 64, padding=0),

            XceptionBlock(64, 128, stride=FRONT_S, first_relu=False),
            XceptionBlock(128, 256, stride=2),
            XceptionBlock(256, 728, stride=2),

            *[XceptionBlock(728, 728) for _ in range(8)],

            XceptionBlock(728, 1024, stride=2, expand_first=False),

            SeparableConv2d(1024, 1536),
            nn.ReLU(inplace=True),
            SeparableConv2d(1536, 2048),
            nn.ReLU(inplace=True)
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
