import os
import torch
import torch.nn as nn
from .core import blocks

__all__ = ['MobileNetMux', 'mobilenet_mux', 'MobileNetMuxv2',
           'mobilenet_mux_v2', 'MobileNetMuxv3', 'mobilenet_mux_v3',
           'MobileNetMuxv4', 'mobilenet_mux_v4']


def mobilenet_mux(pretrained: bool = False, pth: str = None):
    model = MobileNetMux()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetMux(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        filters: int = 32
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filters = filters

        self.features = self.make_layers(filters)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(filters * 32, num_classes)
        )

    def make_layers(self, filters):
        dw1 = blocks.MuxDepthwiseConv2d(filters)
        dw1s2 = blocks.MuxDepthwiseConv2d(filters, stride=2)

        dw2 = blocks.MuxDepthwiseConv2d(filters * 2, mux_layer=dw1)
        dw2s2 = blocks.MuxDepthwiseConv2d(
            filters * 2, stride=2, mux_layer=dw1s2)

        dw4 = blocks.MuxDepthwiseConv2d(filters * 4, mux_layer=dw2)
        dw4s2 = blocks.MuxDepthwiseConv2d(
            filters * 4, stride=2, mux_layer=dw2s2)

        dw8 = blocks.MuxDepthwiseConv2d(filters * 8, mux_layer=dw4)
        dw8s2 = blocks.MuxDepthwiseConv2d(
            filters * 8, stride=2, mux_layer=dw4s2)

        dw16 = blocks.MuxDepthwiseConv2d(
            filters * 16, mux_layer=dw8)
        dw16s2 = blocks.MuxDepthwiseConv2d(
            filters * 16, stride=2, mux_layer=dw8s2)

        return nn.Sequential(
            blocks.Conv2dBlock(self.in_channels, filters * 1, stride=2),

            dw1,
            blocks.PointwiseBlock(filters * 1, filters * 2),

            dw2s2,
            blocks.PointwiseBlock(filters * 2, filters * 4),

            dw4,
            blocks.PointwiseBlock(filters * 4, filters * 4),

            dw4s2,
            blocks.PointwiseBlock(filters * 4, filters * 8),

            dw8,
            blocks.PointwiseBlock(filters * 8, filters * 8),

            dw8s2,
            blocks.PointwiseBlock(filters * 8, filters * 16),

            blocks.MuxDepthwiseConv2d(filters * 16, mux_layer=dw8),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            blocks.MuxDepthwiseConv2d(filters * 16, mux_layer=dw8),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            blocks.MuxDepthwiseConv2d(filters * 16, mux_layer=dw8),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            blocks.MuxDepthwiseConv2d(filters * 16, mux_layer=dw8),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            dw16,
            blocks.PointwiseBlock(filters * 16, filters * 16),

            dw16s2,
            blocks.PointwiseBlock(filters * 16, filters * 32),

            blocks.MuxDepthwiseConv2d(filters * 32, mux_layer=dw16),
            blocks.PointwiseBlock(filters * 32, filters * 32),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_mux_v2(pretrained: bool = False, pth: str = None):
    model = MobileNetMuxv2()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetMuxv2(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1000,
                 filters: int = 32
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.filters = filters

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters * 1, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            blocks.PointwiseBlock(filters * 1, filters * 2),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            blocks.PointwiseBlock(filters * 2, filters * 4),

            blocks.SharedDepthwiseConv2d(filters * 4),
            blocks.PointwiseBlock(filters * 4, filters * 4),

            blocks.SharedDepthwiseConv2d(filters * 4, stride=2),
            blocks.PointwiseBlock(filters * 4, filters * 8),

            blocks.SharedDepthwiseConv2d(filters * 8),
            blocks.PointwiseBlock(filters * 8, filters * 8),

            blocks.SharedDepthwiseConv2d(filters * 8, stride=2),
            blocks.PointwiseBlock(filters * 8, filters * 16),

            blocks.SharedDepthwiseConv2d(filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            blocks.SharedDepthwiseConv2d(filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            blocks.SharedDepthwiseConv2d(filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            blocks.SharedDepthwiseConv2d(filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            blocks.SharedDepthwiseConv2d(filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            blocks.SharedDepthwiseConv2d(filters * 16, stride=2),
            blocks.PointwiseBlock(filters * 16, filters * 32),

            blocks.SharedDepthwiseConv2d(filters * 32, t=4),
            blocks.PointwiseBlock(filters * 32, filters * 32),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(filters * 32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_mux_v3(pretrained: bool = False, pth: str = None):
    model = MobileNetMuxv3()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetMuxv3(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            blocks.PointwiseBlock(filters * 1, filters * 2),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            blocks.PointwiseBlock(filters * 2, filters * 4),

            blocks.SplitIdentityDepthWiseConv2dLayer(filters * 4),
            blocks.PointwiseBlock(filters * 4, filters * 4),

            blocks.SharedDepthwiseConv2d(filters * 4, stride=2, t=16),
            blocks.PointwiseBlock(filters * 4, filters * 8),

            blocks.SplitIdentityDepthWiseConv2dLayer(filters * 8),
            blocks.PointwiseBlock(filters * 8, filters * 8),

            blocks.SharedDepthwiseConv2d(filters * 8, stride=2, t=16),
            blocks.PointwiseBlock(filters * 8, filters * 16),

            blocks.SplitIdentityDepthWiseConv2dLayer(filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            blocks.SplitIdentityDepthWiseConv2dLayer(filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            blocks.SplitIdentityDepthWiseConv2dLayer(filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            blocks.SplitIdentityDepthWiseConv2dLayer(filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            blocks.SharedDepthwiseConv2d(filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            blocks.SharedDepthwiseConv2d(filters * 16, stride=2, t=32),
            blocks.PointwiseBlock(filters * 16, filters * 32),

            blocks.SharedDepthwiseConv2d(filters * 32, t=32),
            blocks.PointwiseBlock(filters * 32, filters * 32),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(filters * 32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_mux_v4(pretrained: bool = False, pth: str = None):
    model = MobileNetMuxv4()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetMuxv4(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.share = blocks.DepthwiseConv2d(filters * 16, filters * 16)

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            blocks.PointwiseBlock(filters, filters * 2),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            blocks.PointwiseBlock(filters * 2, filters * 4),

            blocks.DepthwiseConv2d(filters * 4, filters * 4),
            blocks.PointwiseBlock(filters * 4, filters * 4),

            blocks.DepthwiseConv2d(filters * 4, filters * 4, stride=2),
            blocks.PointwiseBlock(filters * 4, filters * 8),

            blocks.DepthwiseConv2d(filters * 8, filters * 8),
            blocks.PointwiseBlock(filters * 8, filters * 8),

            blocks.DepthwiseConv2d(filters * 8, filters * 8, stride=2),
            blocks.PointwiseBlock(filters * 8, filters * 16),

            self.share,
            blocks.PointwiseBlock(filters * 16, filters * 16),

            self.share,
            blocks.PointwiseBlock(filters * 16, filters * 16),

            self.share,
            blocks.PointwiseBlock(filters * 16, filters * 16),

            self.share,
            blocks.PointwiseBlock(filters * 16, filters * 16),

            self.share,
            blocks.PointwiseBlock(filters * 16, filters * 16),

            blocks.DepthwiseConv2d(filters * 16, filters * 16, stride=2),
            blocks.PointwiseBlock(filters * 16, filters * 32),

            blocks.DepthwiseConv2d(filters * 32, filters * 32),
            blocks.PointwiseBlock(filters * 32, filters * 32),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(filters * 32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
