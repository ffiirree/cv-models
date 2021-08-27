import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Identity
from .core import blocks

__all__ = ['MobileNetMux', 'mobilenet_mux', 'MobileNetMuxv2',
           'mobilenet_mux_v2', 'MobileNetMuxv3', 'mobilenet_mux_v3',
           'MobileNetMuxv4', 'mobilenet_mux_v4', 'mobilenet_mux_v5',
           'mobilenet_mux_v6', 'mobilenet_mux_v7', 'mobilenet_mux_v8',
           'mobilenet_mux_v9', 'mobilenet_mux_v10', 'mobilenet_mux_v11',
           'mobilenet_mux_v12', 'mobilenet_mux_v13', 'mobilenet_mux_v14',
           'mobilenet_mux_v15', 'mobilenet_mux_v16', 'mobilenet_mux_v17',
           'mobilenet_mux_v11_res', 'mobilenet_mux_v9_filter', 'mobilenet_mux_v11_filter']


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

        self.share = blocks.SplitIdentityDepthWiseConv2dLayer(filters * 16)

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


def mobilenet_mux_v5(pretrained: bool = False, pth: str = None):
    model = MobileNetMuxV5()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetMuxV5(nn.Module):
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

            blocks.SharedDepthwiseConv2d(filters * 4, stride=2, t=16),
            blocks.PointwiseBlock(filters * 4, filters * 8),

            blocks.SplitIdentityDepthWiseConv2dLayer(filters * 8),
            blocks.PointwiseBlock(filters * 8, filters * 8),

            blocks.SharedDepthwiseConv2d(filters * 8, stride=2, t=16),
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

            blocks.SharedDepthwiseConv2d(filters * 16, stride=2, t=32),
            blocks.PointwiseBlock(filters * 16, filters * 32),

            blocks.SharedDepthwiseConv2d(filters * 32, t=16),
            blocks.PointwiseBlock(filters * 32, filters * 32),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(filters * 32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_mux_v6(pretrained: bool = False, pth: str = None):
    model = MobileNetMuxV6()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetMuxV6(nn.Module):
    """相较于v5，最后一层"""
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

            blocks.SharedDepthwiseConv2d(filters * 4, stride=2, t=16),
            blocks.PointwiseBlock(filters * 4, filters * 8),

            blocks.SplitIdentityDepthWiseConv2dLayer(filters * 8),
            blocks.PointwiseBlock(filters * 8, filters * 8),

            blocks.SharedDepthwiseConv2d(filters * 8, stride=2, t=16),
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

            blocks.SharedDepthwiseConv2d(filters * 16, stride=2, t=32),
            blocks.PointwiseBlock(filters * 16, filters * 32),

            blocks.SharedDepthwiseConv2d(filters * 32, t=16),
            blocks.PointwiseBlock(filters * 32, filters * 16),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(filters * 16, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SplitIdentityBlock(nn.Module):
    def __init__(
        self,
        inp: int,
        combined: bool = True,
        combine: bool = True
    ):
        super().__init__()

        self.inp = inp // 2

        self.split = blocks.ChannelChunk(2) if combined else nn.Identity()
        self.branch1 = nn.Identity()
        self.branch2 = blocks.DepthwiseConv2d(self.inp, self.inp)
        self.combine1 = blocks.Combine('CONCAT')

        self.pointwise = blocks.PointwiseBlock(inp, inp // 2)
        self.combine2 = blocks.Combine('CONCAT') if combine else nn.Identity()

    def forward(self, x):
        x1, x2 = self.split(x)
        out = self.combine1([self.branch1(x1), self.branch2(x2)])
        out = self.combine2([self.branch1(x2), self.pointwise(out)])
        return out


class SplitIdentityPointwise(nn.Module):
    def __init__(
        self,
        inp: int
    ):
        super().__init__()

        self.inp = inp // 2

        self.split = blocks.ChannelChunk(2)
        self.branch1 = nn.Identity()

        self.branch2 = blocks.PointwiseBlock(inp, inp // 2)
        self.combine = blocks.Combine('CONCAT')

    def forward(self, x):
        x1, _ = self.split(x)
        out = self.combine([self.branch1(x1), self.branch2(x)])
        return out


class SplitIdentityPointwiseX2(nn.Module):
    def __init__(
        self,
        inp: int,
        combine: bool = True
    ):
        super().__init__()

        self.branch1 = nn.Identity()

        self.branch2 = blocks.PointwiseBlock(inp, inp)
        self.combine = blocks.Combine('CONCAT') if combine else nn.Identity()

    def forward(self, x):
        out = self.combine([self.branch1(x), self.branch2(x)])
        return out


def mobilenet_mux_v7(pretrained: bool = False, pth: str = None):
    model = MobileNetMuxV7()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetMuxV7(nn.Module):
    """"""
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            blocks.PointwiseBlock(filters * 1, filters * 2),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            blocks.PointwiseBlock(filters * 2, filters * 4),

            SplitIdentityBlock(filters * 4),

            blocks.SharedDepthwiseConv2d(filters * 4, stride=2, t=16),
            blocks.PointwiseBlock(filters * 4, filters * 8),

            SplitIdentityBlock(filters * 8),

            blocks.SharedDepthwiseConv2d(filters * 8, stride=2, t=16),
            blocks.PointwiseBlock(filters * 8, filters * 16),

            SplitIdentityBlock(filters * 16),

            SplitIdentityBlock(filters * 16),

            SplitIdentityBlock(filters * 16),

            SplitIdentityBlock(filters * 16),

            blocks.SharedDepthwiseConv2d(filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            blocks.SharedDepthwiseConv2d(filters * 16, stride=2, t=32),
            blocks.PointwiseBlock(filters * 16, filters * 32),

            blocks.SharedDepthwiseConv2d(filters * 32, t=32),
            blocks.PointwiseBlock(filters * 32, filters * 16),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(filters * 16, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_mux_v8(pretrained: bool = False, pth: str = None):
    model = MobileNetMuxV8()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetMuxV8(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            blocks.PointwiseBlock(filters, filters * 2),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            SplitIdentityPointwiseX2(filters * 2),

            SplitIdentityBlock(filters * 4),

            # blocks.SharedDepthwiseConv2d(filters * 4, stride=2, t=16),
            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4),

            SplitIdentityBlock(filters * 8),
            SplitIdentityBlock(filters * 8),

            # blocks.SharedDepthwiseConv2d(filters * 8, stride=2, t=16),
            blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8),

            SplitIdentityBlock(filters * 16),
            # SplitIdentityBlock(filters * 16),
            SplitIdentityBlock(filters * 16),
            SplitIdentityBlock(filters * 16),

            blocks.SharedDepthwiseConv2d(filters * 16),
            SplitIdentityPointwise(filters * 16),

            # blocks.SharedDepthwiseConv2d(filters * 16, stride=2, t=32),
            blocks.GaussianFilter(filters * 16, stride=2),
            SplitIdentityPointwiseX2(filters * 16),

            blocks.SharedDepthwiseConv2d(filters * 32, t=32),
            blocks.PointwiseBlock(filters * 32, 512),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_mux_v9(pretrained: bool = False, pth: str = None):
    model = MobileNetMuxV9()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetMuxV9(nn.Module):
    """相较于v7，"""
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 24):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            blocks.PointwiseBlock(filters, filters * 2),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            SplitIdentityPointwiseX2(filters * 2, False),

            SplitIdentityBlock(filters * 4, False),

            blocks.SharedDepthwiseConv2d(filters * 4, stride=2, t=6),
            # blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4, False),

            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False),

            blocks.SharedDepthwiseConv2d(filters * 8, stride=2, t=12),
            # blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8, False),

            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False),

            blocks.SharedDepthwiseConv2d(filters * 16, stride=2, t=24),
            # blocks.GaussianFilter(filters * 16, stride=2),
            SplitIdentityPointwise(filters * 16),

            SplitIdentityBlock(filters * 16, True, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False),

            blocks.SharedDepthwiseConv2d(filters * 16, t=8),
            blocks.PointwiseBlock(filters * 16, 512),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            # nn.Dropout(0.15),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_mux_v10(pretrained: bool = False, pth: str = None):
    model = MobileNetMuxV10()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class SplitIdentityBlockV2(nn.Module):
    def __init__(
        self,
        inp: int
    ):
        super().__init__()

        self.inp = inp // 2

        self.split = blocks.ChannelChunk(2)
        self.branch1 = nn.Identity()
        self.branch2 = blocks.SharedDepthwiseConv2d(self.inp, t=2)
        self.combine1 = blocks.Combine('CONCAT')

        self.pointwise = blocks.PointwiseBlock(inp, inp // 2)
        self.combine2 = blocks.Combine('CONCAT')

    def forward(self, x):
        x1, x2 = self.split(x)
        out = self.combine1([self.branch1(x1), self.branch2(x2)])
        out = self.combine2([self.branch1(x2), self.pointwise(out)])
        return out


class MobileNetMuxV10(nn.Module):
    """Params: 1M"""
    @blocks.batchnorm(position='after')
    # @blocks.nonlinear(nn.SiLU)
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=2),

            SplitIdentityBlock(filters),

            blocks.DepthwiseConv2d(filters, filters, stride=2),
            SplitIdentityPointwiseX2(filters, False),

            SplitIdentityBlock(filters * 2, False),

            blocks.SharedDepthwiseConv2d(filters * 2, stride=2, t=4),
            SplitIdentityPointwiseX2(filters * 2, combine=False),

            # SplitIdentityBlock(filters * 4, input_type='LIST', combine='LIST'),

            SplitIdentityBlock(filters * 4, False),

            blocks.SharedDepthwiseConv2d(filters * 4, stride=2, t=8),
            SplitIdentityPointwiseX2(filters * 4, combine=False),

            SplitIdentityBlock(filters * 8, False, False),

            SplitIdentityBlock(filters * 8, False, False),

            SplitIdentityBlock(filters * 8, False, False),

            SplitIdentityBlock(filters * 8, False),

            blocks.SharedDepthwiseConv2d(filters * 8, stride=2, t=16),
            SplitIdentityPointwiseX2(filters * 8),

            # SplitIdentityBlock(filters * 16, input_type='LIST'),

            blocks.DepthwiseBlock(filters * 16, filters * 16),
            blocks.PointwiseBlock(filters * 16, 498),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            # nn.Dropout(0.15),
            nn.Linear(498, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_mux_v11(pretrained: bool = False, pth: str = None):
    model = MobileNetMuxV11(3, 1000, 32)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetMuxV11(nn.Module):
    """Params: 1M"""
    @blocks.batchnorm(position='after')
    # @blocks.nonlinear(nn.SiLU)
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=2),

            SplitIdentityBlock(filters),

            blocks.DepthwiseConv2d(filters, filters, stride=2),
            SplitIdentityPointwiseX2(filters, False),

            SplitIdentityBlock(filters * 2, False),

            blocks.SharedDepthwiseConv2d(filters * 2, stride=2, t=4),
            # blocks.GaussianFilter(filters * 2, stride=2),
            SplitIdentityPointwiseX2(filters * 2, False),

            # SplitIdentityBlock(filters * 4, False, False),
            SplitIdentityBlock(filters * 4, False),

            blocks.SharedDepthwiseConv2d(filters * 4, stride=2, t=8),
            # blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4, False),

            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False),

            blocks.SharedDepthwiseConv2d(filters * 8, stride=2, t=16),
            # blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8),

            # SplitIdentityBlock(filters * 16, False, False),
            # SplitIdentityBlock(filters * 16, False),

            blocks.SharedDepthwiseConv2d(filters * 16, t=2),
            blocks.PointwiseBlock(filters * 16, 496),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(496, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_mux_v12(pretrained: bool = False, pth: str = None):
    model = MobileNetMuxV12()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetMuxV12(nn.Module):
    """相较于v7，"""
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            SplitIdentityPointwiseX2(filters),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            SplitIdentityPointwiseX2(filters * 2, combine=False),

            SplitIdentityBlock(filters * 4, combined=False),

            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4, combine=False),

            SplitIdentityBlock(filters * 8, combined=False, combine=False),
            SplitIdentityBlock(filters * 8, combined=False),

            blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8, combine=False),

            SplitIdentityBlock(filters * 16, combined=False, combine=False),
            SplitIdentityBlock(filters * 16, combined=False),

            blocks.GaussianFilter(filters * 16, stride=2),
            SplitIdentityPointwise(filters * 16),

            SplitIdentityBlock(filters * 16),

            blocks.DepthwiseConv2d(filters * 16, filters * 16),
            blocks.PointwiseBlock(filters * 16, 520),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(520, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_mux_v13(pretrained: bool = False, pth: str = None):
    model = MobileNetMuxV13()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetMuxV13(nn.Module):
    """相较于v7，"""
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            blocks.PointwiseBlock(filters, filters * 2),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            SplitIdentityPointwiseX2(filters * 2),

            SplitIdentityBlock(filters * 4),

            # blocks.SharedDepthwiseConv2d(filters * 4, stride=2, t=16),
            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4),

            SplitIdentityBlock(filters * 8),

            # blocks.SharedDepthwiseConv2d(filters * 8, stride=2, t=16),
            blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8),

            SplitIdentityBlock(filters * 16),
            SplitIdentityBlock(filters * 16),
            SplitIdentityBlock(filters * 16),
            SplitIdentityBlock(filters * 16),

            blocks.SharedDepthwiseConv2d(filters * 16),
            SplitIdentityPointwise(filters * 16),

            # blocks.SharedDepthwiseConv2d(filters * 16, stride=2, t=32),
            blocks.GaussianFilter(filters * 16, stride=2),
            SplitIdentityPointwiseX2(filters * 16),

            blocks.SharedDepthwiseConv2d(filters * 32, t=16),
            blocks.PointwiseBlock(filters * 32, filters * 16),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(filters * 16, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_mux_v14(pretrained: bool = False, pth: str = None):
    model = MobileNetMuxV14()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetMuxV14(nn.Module):
    """相较于v7，"""
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            SplitIdentityPointwiseX2(filters),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            SplitIdentityPointwiseX2(filters * 2, combine=False),

            SplitIdentityBlock(filters * 4, combined=False),

            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4, combine=False),

            SplitIdentityBlock(filters * 8, combined=False, combine=False),
            SplitIdentityBlock(filters * 8, combined=False),

            blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8, combine=False),

            SplitIdentityBlock(filters * 16, combined=False, combine=False),
            SplitIdentityBlock(filters * 16, combined=False, combine=False),
            SplitIdentityBlock(filters * 16, combined=False, combine=False),
            SplitIdentityBlock(filters * 16, combined=False),

            blocks.GaussianFilter(filters * 16, stride=2),
            SplitIdentityPointwiseX2(filters * 16),

            SplitIdentityBlock(filters * 32),

            blocks.DepthwiseConv2d(filters * 32, filters * 32),
            blocks.PointwiseBlock(filters * 32, filters * 32),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(filters * 32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_mux_v15(pretrained: bool = False, pth: str = None):
    model = MobileNetMuxV15()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetMuxV15(nn.Module):
    """相较于v7，"""
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            SplitIdentityPointwiseX2(filters),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            SplitIdentityPointwiseX2(filters * 2, combine=False),

            SplitIdentityBlock(filters * 4, combined=False),

            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4, combine=False),

            SplitIdentityBlock(filters * 8, combined=False, combine=False),
            SplitIdentityBlock(filters * 8, combined=False, combine=False),
            SplitIdentityBlock(filters * 8, combined=False, combine=False),
            SplitIdentityBlock(filters * 8, combined=False),

            blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8, combine=False),

            SplitIdentityBlock(filters * 16, combined=False, combine=False),
            SplitIdentityBlock(filters * 16, combined=False, combine=False),
            SplitIdentityBlock(filters * 16, combined=False, combine=False),
            SplitIdentityBlock(filters * 16, combined=False, combine=False),
            SplitIdentityBlock(filters * 16, combined=False, combine=False),
            SplitIdentityBlock(filters * 16, combined=False),

            blocks.GaussianFilter(filters * 16, stride=2),
            SplitIdentityPointwiseX2(filters * 16),

            SplitIdentityBlock(filters * 32, combine=False),
            SplitIdentityBlock(filters * 32, combined=False, combine=False),
            SplitIdentityBlock(filters * 32, combined=False),

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


def mobilenet_mux_v16(pretrained: bool = False, pth: str = None):
    model = MobileNetMuxV16()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetMuxV16(nn.Module):
    """相较于v7，"""
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            SplitIdentityPointwiseX2(filters),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            SplitIdentityPointwiseX2(filters * 2, combine=False),

            SplitIdentityBlock(filters * 4, combined=False, combine=False),
            SplitIdentityBlock(filters * 4, combined=False, combine=False),
            SplitIdentityBlock(filters * 4, combined=False),

            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4, combine=False),

            SplitIdentityBlock(filters * 8, combined=False, combine=False),
            SplitIdentityBlock(filters * 8, combined=False, combine=False),
            SplitIdentityBlock(filters * 8, combined=False, combine=False),
            SplitIdentityBlock(filters * 8, combined=False),

            blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8, combine=False),

            SplitIdentityBlock(filters * 16, combined=False, combine=False),
            SplitIdentityBlock(filters * 16, combined=False, combine=False),
            SplitIdentityBlock(filters * 16, combined=False, combine=False),
            SplitIdentityBlock(filters * 16, combined=False, combine=False),
            SplitIdentityBlock(filters * 16, combined=False, combine=False),
            SplitIdentityBlock(filters * 16, combined=False),

            blocks.GaussianFilter(filters * 16, stride=2),
            SplitIdentityPointwiseX2(filters * 16),

            SplitIdentityBlock(filters * 32, combine=False),
            SplitIdentityBlock(filters * 32, combined=False, combine=False),
            SplitIdentityBlock(filters * 32, combined=False, combine=False),
            SplitIdentityBlock(filters * 32, combined=False, combine=False),
            SplitIdentityBlock(filters * 32, combined=False),

            blocks.DepthwiseConv2d(filters * 32, filters * 32),
            blocks.PointwiseBlock(filters * 32, filters * 32),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(filters * 32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_mux_v17(pretrained: bool = False, pth: str = None):
    model = MobileNetMuxV17()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetMuxV17(nn.Module):
    """相较于v7，"""
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            SplitIdentityPointwiseX2(filters),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            SplitIdentityPointwiseX2(filters * 2, combine=False),

            SplitIdentityBlock(filters * 4, combined=False, combine=False),
            SplitIdentityBlock(filters * 4, combined=False, combine=False),
            SplitIdentityBlock(filters * 4, combined=False, combine=False),
            SplitIdentityBlock(filters * 4, combined=False),

            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4, combine=False),

            SplitIdentityBlock(filters * 8, combined=False, combine=False),
            SplitIdentityBlock(filters * 8, combined=False, combine=False),
            SplitIdentityBlock(filters * 8, combined=False, combine=False),
            SplitIdentityBlock(filters * 8, combined=False, combine=False),
            SplitIdentityBlock(filters * 8, combined=False, combine=False),
            SplitIdentityBlock(filters * 8, combined=False),

            blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8, combine=False),

            # SplitIdentityBlock(filters * 16, combined=False, combine=False),
            SplitIdentityBlock(filters * 16, combined=False, combine=False),
            SplitIdentityBlock(filters * 16, combined=False, combine=False),
            SplitIdentityBlock(filters * 16, combined=False, combine=False),
            SplitIdentityBlock(filters * 16, combined=False, combine=False),
            SplitIdentityBlock(filters * 16, combined=False, combine=False),
            SplitIdentityBlock(filters * 16, combined=False, combine=False),
            SplitIdentityBlock(filters * 16, combined=False, combine=False),
            SplitIdentityBlock(filters * 16, combined=False),

            blocks.GaussianFilter(filters * 16, stride=2),
            SplitIdentityPointwiseX2(filters * 16),

            SplitIdentityBlock(filters * 32, combine=False),
            SplitIdentityBlock(filters * 32, combined=False, combine=False),
            SplitIdentityBlock(filters * 32, combined=False, combine=False),
            SplitIdentityBlock(filters * 32, combined=False, combine=False),
            SplitIdentityBlock(filters * 32, combined=False, combine=False),
            SplitIdentityBlock(filters * 32, combined=False, combine=False),
            SplitIdentityBlock(filters * 32, combined=False, combine=False),
            SplitIdentityBlock(filters * 32, combined=False, combine=False),
            SplitIdentityBlock(filters * 32, combined=False, combine=False),
            SplitIdentityBlock(filters * 32, combined=False, combine=False),
            SplitIdentityBlock(filters * 32, combined=False),

            blocks.DepthwiseConv2d(filters * 32, filters * 32),
            blocks.PointwiseBlock(filters * 32, 1280),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_mux_v11_res(pretrained: bool = False, pth: str = None):
    model = MobileNetMuxV11Res()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class SplitIdentityBlockRes(nn.Module):
    def __init__(
        self,
        inp: int
    ):
        super().__init__()

        self.branch1 = nn.Identity()
        self.branch2 = blocks.DepthwiseConv2d(inp, inp)
        self.cat = blocks.Combine('CONCAT')

        self.pointwise = blocks.PointwiseBlock(inp * 2, inp)
        self.add = blocks.Combine('ADD')

    def forward(self, x):
        identity = x
        out = self.cat([self.branch1(x), self.branch2(x)])

        out = self.add([self.branch1(identity), self.pointwise(out)])
        return out


class MobileNetMuxV11Res(nn.Module):
    """Params: 1M"""
    @blocks.batchnorm(position='after')
    # @blocks.nonlinear(nn.SiLU)
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=2),

            # SplitIdentityBlockRes(filters),

            blocks.DepthwiseConv2d(filters, filters, stride=2),
            blocks.PointwiseBlock(filters, filters),

            SplitIdentityBlockRes(filters),

            blocks.GaussianFilter(filters, stride=2),
            SplitIdentityPointwiseX2(filters),

            # SplitIdentityBlockRes(filters * 2),
            SplitIdentityBlockRes(filters * 2),

            blocks.GaussianFilter(filters * 2, stride=2),
            SplitIdentityPointwiseX2(filters * 2),

            SplitIdentityBlockRes(filters * 4),
            SplitIdentityBlockRes(filters * 4),
            SplitIdentityBlockRes(filters * 4),
            SplitIdentityBlockRes(filters * 4),

            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4),

            # SplitIdentityBlockRes(filters * 8),
            # SplitIdentityBlockRes(filters * 8),
            # SplitIdentityBlockRes(filters * 8),

            blocks.DepthwiseBlock(filters * 8, filters * 8),
            SplitIdentityPointwiseX2(filters * 8),

            blocks.DepthwiseBlock(filters * 16, filters * 16),
            blocks.PointwiseBlock(filters * 16, 496),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(496, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_mux_v9_filter(pretrained: bool = False, pth: str = None):
    model = MobileNetMuxV9Filter()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class Filter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stride: int = 1,
        dilation: int = 1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.kernel_size = (3, 3)
        self.padding = (1, 1)
        self.stride = (stride, stride)
        self.dilation = (dilation, dilation)
        self.groups = in_channels
        self.padding_mode = 'zeros'

        sharpness = torch.tensor(
            [[[
                [-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]
            ]], [[
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ]], [[
                [-1, 0, -1],
                [0, 5, 0],
                [-1, 0, -1]
            ]]], dtype=torch.float32
        )

        edge = torch.tensor(
            [[[
                [-1, -1, -1],
                [0,  0,  0],
                [1,  1,  1]
            ]], [[
                [-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1]
            ]], [[
                [-1/3, -2/3, -1],
                [0, 0, 0],
                [1/3, 2/3, 1]
            ]], [[
                [-1/2, 0, 1/2],
                [-2/2, 0, 2/2],
                [-1/2, 0, 1/2]
            ]], [[
                [-1/2, 0, 0],
                [0, 2/2, 0],
                [0, 0, -1/2]
            ]], [[
                [0, 0, -1/2],
                [0, 2/2, 0],
                [-1/2, 0, 0]
            ]], [[
                [0, -1/2, 0],
                [0,  2/2, 0],
                [0, -1/2, 0]
            ]], [[
                [0, 0, 0],
                [-1/2,  2/2, -1/2],
                [0, 0, 0]
            ]], [[
                [0, -1/2, 0],
                [0, -1/2, 0],
                [0, 2/2, 0]
            ]], [[
                [0, 0, 0],
                [-1/2, -1/2, 2/2],
                [0, 0, 0]
            ]], [[
                [-1/8, -1/8, -1/8],
                [-1/8, 8/8, -1/8],
                [-1/8, -1/8, -1/8]
            ]]], dtype=torch.float32
        )

        embossing = torch.tensor(
            [[[
                [-1, -1, 0],
                [-1, 0, 1],
                [0, 1, 1]
            ]], [[
                [0, 1, 1],
                [-1, 0, 1],
                [-1, -1, 0]
            ]], [[
                [-3/3, -2/3, -1/3],
                [-2/3, 0, 2/3],
                [1/3, 2/3, 3/3]
            ]], [[
                [0, -1, 0],
                [-1, 0, 1],
                [0, 1, 0]
            ]]], dtype=torch.float32
        )

        box = torch.tensor(
            [[[
                [1/9, 1/9, 1/9],
                [1/9, 1/9, 1/9],
                [1/9, 1/9, 1/9]
            ]], [[
                [.0, 1/5, .0],
                [1/5, 1/5, 1/5],
                [.0, 1/5, .0]
            ]]], dtype=torch.float32
        )

        gaussian = torch.tensor([[[
            [0.0811, 0.1226, 0.0811],
            [0.1226, 0.1853, 0.1226],
            [0.0811, 0.1226, 0.0811]
        ]], [[
            [0.0571, 0.1248, 0.0571],
            [0.1248, 0.2725, 0.1248],
            [0.0571, 0.1248, 0.0571]
        ]], [[
            [0.0439, 0.1217, 0.0439],
            [0.1217, 0.3377, 0.1217],
            [0.0439, 0.1217, 0.0439]
        ]], [[
            [0.0277, 0.1110, 0.0277],
            [0.1110, 0.4452, 0.1110],
            [0.0277, 0.1110, 0.0277]
        ]]], dtype=torch.float32)

        # motion = torch.tensor(
        #     [[[
        #         [1/3, 0, 0],
        #         [0, 1/3, 0],
        #         [0, 0, 1/3]
        #     ]]], dtype=torch.float32
        # )

        kernels = torch.cat(
            [sharpness, edge, embossing, box, gaussian], dim=0)

        self.weight = nn.Parameter(kernels.repeat(
            self.in_channels // 24, 1, 1, 1), False)
        self.register_parameter('bias', None)

        self.weight.requires_grad_(False)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)


class SplitIdentityBlockFilter(nn.Module):
    def __init__(
        self,
        inp: int,
        combined: bool = True,
        combine: bool = True
    ):
        super().__init__()

        self.inp = inp // 2

        self.split = blocks.ChannelChunk(2) if combined else nn.Identity()
        self.branch1 = nn.Identity()
        self.branch2 = Filter(self.inp)
        self.combine1 = blocks.Combine('CONCAT')

        self.pointwise = blocks.PointwiseBlock(inp, inp // 2)
        self.combine2 = blocks.Combine('CONCAT') if combine else nn.Identity()

    def forward(self, x):
        x1, x2 = self.split(x)
        out = self.combine1([self.branch1(x1), self.branch2(x2)])
        out = self.combine2([self.branch1(x2), self.pointwise(out)])
        return out


class MobileNetMuxV9Filter(nn.Module):
    """相较于v7，"""
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 24):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            blocks.PointwiseBlock(filters, filters * 2),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            SplitIdentityPointwiseX2(filters * 2, False),

            SplitIdentityBlockFilter(filters * 4, False),

            # blocks.SharedDepthwiseConv2d(filters * 4, stride=2, t=16),
            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4, False),

            SplitIdentityBlockFilter(filters * 8, False, False),
            SplitIdentityBlockFilter(filters * 8, False, False),
            SplitIdentityBlockFilter(filters * 8, False),

            # blocks.SharedDepthwiseConv2d(filters * 8, stride=2, t=32),
            blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8, False),

            SplitIdentityBlockFilter(filters * 16, False, False),
            SplitIdentityBlockFilter(filters * 16, False, False),
            SplitIdentityBlockFilter(filters * 16, False, False),
            SplitIdentityBlockFilter(filters * 16, False, False),
            SplitIdentityBlockFilter(filters * 16, False),

            # blocks.SharedDepthwiseConv2d(filters * 16, stride=2, t=32),
            blocks.GaussianFilter(filters * 16, stride=2),
            SplitIdentityPointwise(filters * 16),

            SplitIdentityBlockFilter(filters * 16, True, False),
            SplitIdentityBlockFilter(filters * 16, False, False),
            SplitIdentityBlockFilter(filters * 16, False),

            blocks.SharedDepthwiseConv2d(filters * 16, t=8),
            blocks.PointwiseBlock(filters * 16, 512),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            # nn.Dropout(0.15),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_mux_v11_filter(pretrained: bool = False, pth: str = None):
    model = MobileNetMuxV11Filter(3, 1000, 32)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class Filters32(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stride: int = 1,
        dilation: int = 1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.kernel_size = (3, 3)
        self.padding = (1, 1)
        self.stride = (stride, stride)
        self.dilation = (dilation, dilation)
        self.groups = in_channels
        self.padding_mode = 'zeros'

        sharpness = torch.tensor(
            [[[
                [-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]
            ]], [[
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ]], [[
                [-1, 0, -1],
                [0, 5, 0],
                [-1, 0, -1]
            ]]], dtype=torch.float32
        )

        edge = torch.tensor(
            [[[
                [-1, -1, -1],
                [0,  0,  0],
                [1,  1,  1]
            ]], [[
                [-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1]
            ]], [[
                [-1/2, -2/2, -1/2],
                [0, 0, 0],
                [1/2, 2/2, 1/2]
            ]], [[
                [-1/2, 0, 1/2],
                [-2/2, 0, 2/2],
                [-1/2, 0, 1/2]
            ]], [[
                [-1/2, 0, 0],
                [0, 2/2, 0],
                [0, 0, -1/2]
            ]], [[
                [0, 0, -1/2],
                [0, 2/2, 0],
                [-1/2, 0, 0]
            ]], [[
                [0, -1/2, 0],
                [0,  2/2, 0],
                [0, -1/2, 0]
            ]], [[
                [0, 0, 0],
                [-1/2,  2/2, -1/2],
                [0, 0, 0]
            ]], [[
                [0, -1/2, 0],
                [0, -1/2, 0],
                [0, 2/2, 0]
            ]], [[
                [0, 2/2, 0],
                [0, -1/2, 0],
                [0, -1/2, 0]
            ]], [[
                [0, 0, 0],
                [-1/2, -1/2, 2/2],
                [0, 0, 0]
            ]], [[
                [0, 0, 0],
                [2/2, -1/2, -1/2],
                [0, 0, 0]
            ]], [[
                [-1/8, -1/8, -1/8],
                [-1/8, 8/8, -1/8],
                [-1/8, -1/8, -1/8]
            ]], [[
                [-1, 0, 1],
                [0,  0,  0],
                [1,  0,  -1]
            ]]], dtype=torch.float32
        )

        embossing = torch.tensor(
            [[[
                [-1, -1, 0],
                [-1, 0, 1],
                [0, 1, 1]
            ]], [[
                [0, 1, 1],
                [-1, 0, 1],
                [-1, -1, 0]
            ]], [[
                [-3/3, -2/3, -1/3],
                [-2/3, 0, 2/3],
                [1/3, 2/3, 3/3]
            ]], [[
                [-1/3, -2/3, -3/3],
                [2/3, 0, -2/3],
                [3/3, 2/3, 1/3]
            ]], [[
                [0, -1, 0],
                [-1, 0, 1],
                [0, 1, 0]
            ]], [[
                [0, -1, 0],
                [1, 0, -1],
                [0, 1, 0]
            ]]], dtype=torch.float32
        )

        box = torch.tensor(
            [[[
                [1/9, 1/9, 1/9],
                [1/9, 1/9, 1/9],
                [1/9, 1/9, 1/9]
            ]], [[
                [.0, 1/5, .0],
                [1/5, 1/5, 1/5],
                [.0, 1/5, .0]
            ]], [[
                [1/5, 0, 1/5],
                [0, 1/5, 0],
                [1/5, 0, 1/5]
            ]]], dtype=torch.float32
        )

        gaussian = torch.tensor([[[
            [0.0811, 0.1226, 0.0811],
            [0.1226, 0.1853, 0.1226],
            [0.0811, 0.1226, 0.0811]
        ]], [[
            [0.0571, 0.1248, 0.0571],
            [0.1248, 0.2725, 0.1248],
            [0.0571, 0.1248, 0.0571]
        ]], [[
            [0.0439, 0.1217, 0.0439],
            [0.1217, 0.3377, 0.1217],
            [0.0439, 0.1217, 0.0439]
        ]], [[
            [0.0277, 0.1110, 0.0277],
            [0.1110, 0.4452, 0.1110],
            [0.0277, 0.1110, 0.0277]
        ]]], dtype=torch.float32)

        motion = torch.tensor(
            [[[
                [1/3, 0, 0],
                [0, 1/3, 0],
                [0, 0, 1/3]
            ]], [[
                [0, 0, 1/3],
                [0, 1/3, 0],
                [1/3, 0, 0]
            ]]], dtype=torch.float32
        )

        kernels = torch.cat(
            [sharpness, edge, embossing, box, gaussian, motion], dim=0)

        self.weight = nn.Parameter(kernels.repeat(
            self.in_channels // 32, 1, 1, 1), False)
        self.register_parameter('bias', None)

        self.weight.requires_grad_(False)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)


class SplitIdentityBlockFilters32(nn.Module):
    def __init__(
        self,
        inp: int,
        combined: bool = True,
        combine: bool = True
    ):
        super().__init__()

        self.inp = inp // 2

        self.split = blocks.ChannelChunk(2) if combined else nn.Identity()
        self.branch1 = nn.Identity()
        self.branch2 = Filters32(self.inp)
        self.combine1 = blocks.Combine('CONCAT')

        self.pointwise = blocks.PointwiseBlock(inp, inp // 2)
        self.combine2 = blocks.Combine('CONCAT') if combine else nn.Identity()

    def forward(self, x):
        x1, x2 = self.split(x)
        out = self.combine1([self.branch1(x1), self.branch2(x2)])
        out = self.combine2([self.branch1(x2), self.pointwise(out)])
        return out


class MobileNetMuxV11Filter(nn.Module):
    """Params: 1M"""
    @blocks.batchnorm(position='after')
    # @blocks.nonlinear(nn.SiLU)
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=2),

            SplitIdentityBlock(filters),

            blocks.DepthwiseConv2d(filters, filters, stride=2),
            SplitIdentityPointwiseX2(filters, False),

            SplitIdentityBlockFilters32(filters * 2, False),

            blocks.GaussianFilter(filters * 2, stride=2),
            SplitIdentityPointwiseX2(filters * 2, False),

            SplitIdentityBlockFilters32(filters * 4, False),

            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4, False),

            SplitIdentityBlockFilters32(filters * 8, False, False),
            SplitIdentityBlockFilters32(filters * 8, False, False),
            SplitIdentityBlockFilters32(filters * 8, False, False),
            SplitIdentityBlockFilters32(filters * 8, False),

            blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8),

            blocks.DepthwiseBlock(filters * 16, filters * 16),
            blocks.PointwiseBlock(filters * 16, 496),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(496, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
