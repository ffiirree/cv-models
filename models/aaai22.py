import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.linear import Identity
from .core import blocks

__all__ = ['mobilenet_g1', 'mobilenet_g2', 'mobilenet_g3',
           'mobilenet_g4', 'mobilenet_g4_all', 'mobilenet_g5',
           'mobilenet_g6',
           'mobilenet_g1235', 'mobilenet_g12356',
           'micronet_a1_0', 'micronet_b1_0', 'micronet_c1_0',
           'micronet_a1_5', 'micronet_b1_5', 'micronet_c1_5', 'mobilenet_g7',
           'micronet_b2_0', 'micronet_c2_0', 'micronet_b2_5', 'micronet_c2_5', 'micronet_b5_0', 'micronet_d2_0',
           'threepathnet_x2_0', 'threepathnet_x1_5', 'threepathnet_v2_x1_5']


# class ThreePathBlock(nn.Module):
#     def __init__(
#         self,
#         inp: int,
#         combined: bool = True,
#         combine: bool = True
#     ):
#         super().__init__()

#         self.split = blocks.ChannelChunk(3) if combined else nn.Identity()
#         self.branch1 = nn.Identity()
#         self.branch2 = blocks.DepthwiseConv2d(inp // 3, inp // 3)
#         self.combine1 = blocks.Combine('CONCAT')

#         self.pointwise = blocks.PointwiseBlock(inp, inp // 3)
#         self.combine2 = blocks.Combine('CONCAT') if combine else nn.Identity()

#     def forward(self, x):
#         x1, x2, x3 = self.split(x)
#         out = self.combine1(
#             [self.branch1(x1), self.branch1(x2), self.branch2(x3)])
#         out = self.combine2(
#             [self.branch1(x1), self.branch1(x3), self.pointwise(out)])
#         return out

class ThreePathBlock(nn.Module):
    def __init__(
        self,
        inp: int,
        combined: bool = True,
        combine: bool = True
    ):
        super().__init__()

        self.split = blocks.ChannelChunk(3)
        self.branch1 = nn.Identity()
        self.branch2 = blocks.DepthwiseConv2d(inp // 3, inp // 3)
        self.combine1 = blocks.Combine('CONCAT')

        self.pointwise = nn.Sequential(
            blocks.PointwiseConv2d(inp, inp // 3),
            nn.ReLU(inplace=True)
        )
        self.combine2 = blocks.Combine('CONCAT')
        self.bn = BatchNorm2d(inp)

    def forward(self, x):
        x1, x2, x3 = self.split(x)
        out = self.combine1(
            [self.branch1(x1), self.branch1(x2), self.branch2(x3)])
        out = self.combine2(
            [self.branch1(x1), self.branch1(x3), self.pointwise(out)])
        out = self.bn(out)
        return out


class ThreePathBlockV2(nn.Module):
    def __init__(
        self,
        inp: int,
        combined: bool = True,
        combine: bool = True
    ):
        super().__init__()

        self.split = blocks.ChannelChunk(3) if combined else nn.Identity()
        # self.branch1 = nn.Identity()
        self.depthwise = blocks.DepthwiseConv2d((inp // 3) * 2, (inp // 3) * 2)
        self.cat = blocks.Combine('CONCAT')

        self.pointwise = blocks.PointwiseBlock(inp, inp // 3)
        self.combine = blocks.Combine('CONCAT') if combine else nn.Identity()

    def forward(self, x):
        x1, x2, x3 = self.split(x)
        out = self.cat([x1, self.depthwise(self.cat([x2, x3]))])
        out = self.combine([x3, x1, self.pointwise(out)])
        return out


class TwoPathX2(nn.Module):
    def __init__(
        self,
        inp: int,
        combine: bool = True
    ):
        super().__init__()

        self.branch1 = nn.Identity()

        self.branch2 = nn.Sequential(
            blocks.PointwiseConv2d(inp, inp),
            nn.ReLU(inplace=True)
        )
        self.combine = blocks.Combine('CONCAT')
        self.bn = BatchNorm2d(inp * 2)

    def forward(self, x):
        out = self.combine([self.branch1(x), self.branch2(x)])
        out = self.bn(out)
        return out


def threepathnet_x1_5(pretrained: bool = False, pth: str = None):
    model = ThreePathNet()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class ThreePathNet(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        filters: int = 18
    ):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            TwoPathX2(filters),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            TwoPathX2(filters * 2),

            ThreePathBlock(filters * 4),

            blocks.GaussianFilter(filters * 4, stride=2),
            TwoPathX2(filters * 4),

            ThreePathBlock(filters * 8, True, False),
            ThreePathBlock(filters * 8, False, False),
            ThreePathBlock(filters * 8, False),

            blocks.GaussianFilter(filters * 8, stride=2),
            TwoPathX2(filters * 8),

            ThreePathBlock(filters * 16, True, False),
            ThreePathBlock(filters * 16, False, False),
            ThreePathBlock(filters * 16, False, False),
            ThreePathBlock(filters * 16, False, False),
            ThreePathBlock(filters * 16, False, False),
            ThreePathBlock(filters * 16, False, False),
            ThreePathBlock(filters * 16, False),

            blocks.GaussianFilter(filters * 16, stride=2),
            TwoPathX2(filters * 16),

            ThreePathBlock(filters * 32, True, False),
            ThreePathBlock(filters * 32, False, False),
            ThreePathBlock(filters * 32, False),

            blocks.SharedDepthwiseConv2d(filters * 32),
            blocks.PointwiseBlock(filters * 32, 512),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def threepathnet_v2_x1_5(pretrained: bool = False, pth: str = None):
    model = ThreePathNetV2()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class ThreePathNetV2(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        filters: int = 18
    ):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            SplitIdentityPointwiseX2(filters),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            SplitIdentityPointwiseX2(filters * 2),

            ThreePathBlockV2(filters * 4),

            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4),

            ThreePathBlockV2(filters * 8, True, False),
            ThreePathBlockV2(filters * 8, False, False),
            ThreePathBlockV2(filters * 8, False),

            blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8),

            ThreePathBlockV2(filters * 16, True, False),
            ThreePathBlockV2(filters * 16, False, False),
            ThreePathBlockV2(filters * 16, False, False),
            ThreePathBlockV2(filters * 16, False, False),
            ThreePathBlockV2(filters * 16, False, False),
            ThreePathBlockV2(filters * 16, False, False),
            ThreePathBlockV2(filters * 16, False),

            blocks.GaussianFilter(filters * 16, stride=2),
            SplitIdentityPointwiseX2(filters * 16),

            ThreePathBlockV2(filters * 32, True, False),
            ThreePathBlockV2(filters * 32, False, False),
            ThreePathBlockV2(filters * 32, False),

            blocks.SharedDepthwiseConv2d(filters * 32),
            blocks.PointwiseBlock(filters * 32, 512),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def threepathnet_x2_0(pretrained: bool = False, pth: str = None):
    model = ThreePathNetX2_0()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class ThreePathNetX2_0(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        filters: int = 18
    ):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            TwoPathX2(filters),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            TwoPathX2(filters * 2),

            ThreePathBlock(filters * 4),
            ThreePathBlock(filters * 4),

            blocks.GaussianFilter(filters * 4, stride=2),
            TwoPathX2(filters * 4),

            ThreePathBlock(filters * 8, True, False),
            ThreePathBlock(filters * 8, False, False),
            ThreePathBlock(filters * 8, False, False),
            ThreePathBlock(filters * 8, False, False),
            ThreePathBlock(filters * 8, False),

            blocks.GaussianFilter(filters * 8, stride=2),
            TwoPathX2(filters * 8),

            ThreePathBlock(filters * 16, True, False),
            ThreePathBlock(filters * 16, False, False),
            ThreePathBlock(filters * 16, False, False),
            ThreePathBlock(filters * 16, False, False),
            ThreePathBlock(filters * 16, False, False),
            ThreePathBlock(filters * 16, False, False),
            # ThreePathBlock(filters * 16, False, False),
            # ThreePathBlock(filters * 16, False, False),
            ThreePathBlock(filters * 16, False, False),
            ThreePathBlock(filters * 16, False),

            blocks.GaussianFilter(filters * 16, stride=2),
            TwoPathX2(filters * 16),

            ThreePathBlock(filters * 32, True, False),
            ThreePathBlock(filters * 32, False, False),
            ThreePathBlock(filters * 32, False, False),
            ThreePathBlock(filters * 32, False, False),
            ThreePathBlock(filters * 32, False, False),
            ThreePathBlock(filters * 32, False, False),
            ThreePathBlock(filters * 32, False),

            blocks.SharedDepthwiseConv2d(filters * 32),
            blocks.PointwiseBlock(filters * 32, 512),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_g1(pretrained: bool = False, pth: str = None):
    model = MobileNetG1()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetG1(nn.Module):
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

            blocks.DepthwiseConv2d(filters * 4, filters * 4),
            blocks.PointwiseBlock(filters * 4, filters * 4),

            # blocks.DepthwiseConv2d(filters * 4, filters * 4, stride=2),
            blocks.GaussianFilter(filters * 4, stride=2),
            blocks.PointwiseBlock(filters * 4, filters * 8),

            blocks.DepthwiseConv2d(filters * 8, filters * 8),
            blocks.PointwiseBlock(filters * 8, filters * 8),

            # blocks.DepthwiseConv2d(filters * 8, filters * 8, stride=2),
            blocks.GaussianFilter(filters * 8, stride=2),
            blocks.PointwiseBlock(filters * 8, filters * 16),

            blocks.DepthwiseConv2d(filters * 16, filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            blocks.DepthwiseConv2d(filters * 16, filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            blocks.DepthwiseConv2d(filters * 16, filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            blocks.DepthwiseConv2d(filters * 16, filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            blocks.DepthwiseConv2d(filters * 16, filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            # blocks.DepthwiseConv2d(filters * 16, filters * 16, stride=2),
            blocks.GaussianFilter(filters * 16, stride=2),
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


def mobilenet_g2(pretrained: bool = False, pth: str = None):
    model = MobileNetG2()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetG2(nn.Module):
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

            blocks.DepthwiseConv2d(filters * 4, filters * 4, stride=2),
            blocks.PointwiseBlock(filters * 4, filters * 8),

            blocks.SplitIdentityDepthWiseConv2dLayer(filters * 8),
            blocks.PointwiseBlock(filters * 8, filters * 8),

            blocks.DepthwiseConv2d(filters * 8, filters * 8, stride=2),
            blocks.PointwiseBlock(filters * 8, filters * 16),

            blocks.SplitIdentityDepthWiseConv2dLayer(filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            blocks.SplitIdentityDepthWiseConv2dLayer(filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            blocks.SplitIdentityDepthWiseConv2dLayer(filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            blocks.SplitIdentityDepthWiseConv2dLayer(filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            blocks.SplitIdentityDepthWiseConv2dLayer(filters * 16),
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


# mobilenet_mux_v2
def mobilenet_g3(pretrained: bool = False, pth: str = None):
    model = MobileNetG3()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetG3(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

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

# mobilenet_mux


def mobilenet_g4(pretrained: bool = False, pth: str = None):
    model = MobileNetG4()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetG4(nn.Module):
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

# mobilenet_mux_v4


def mobilenet_g4_all(pretrained: bool = False, pth: str = None):
    model = MobileNetG4All()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetG4All(nn.Module):
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


class SkipBlock(nn.Module):
    def __init__(
        self,
        inp: int
    ):
        super().__init__()

        self.split = blocks.ChannelChunk(2)
        self.branch1 = nn.Identity()
        self.branch2 = nn.Sequential(
            blocks.DepthwiseConv2d(inp, inp),
            blocks.PointwiseBlock(inp, inp // 2)
        )

        self.combine = blocks.Combine('CONCAT')

    def forward(self, x):
        _, x2 = self.split(x)
        out = self.combine([self.branch1(x2), self.branch2(x)])
        return out


def mobilenet_g5(pretrained: bool = False, pth: str = None):
    model = MobileNetG5()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetG5(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters * 1, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            SplitIdentityPointwiseX2(filters),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            SplitIdentityPointwiseX2(filters * 2),

            SkipBlock(filters * 4),

            blocks.DepthwiseConv2d(filters * 4, filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4),

            SkipBlock(filters * 8),

            blocks.DepthwiseConv2d(filters * 8, filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8),

            SkipBlock(filters * 16),
            SkipBlock(filters * 16),
            SkipBlock(filters * 16),
            SkipBlock(filters * 16),
            SkipBlock(filters * 16),

            blocks.DepthwiseConv2d(filters * 16, filters * 16, stride=2),
            SplitIdentityPointwiseX2(filters * 16),

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


def mobilenet_g1235(pretrained: bool = False, pth: str = None):
    model = MobileNetG1235()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetG1235(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters * 1, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            SplitIdentityPointwiseX2(filters),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            SplitIdentityPointwiseX2(filters * 2, False),

            SplitIdentityBlock(filters * 4, False),

            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4, False),

            SplitIdentityBlock(filters * 8, False),

            blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8, False),

            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False),

            blocks.GaussianFilter(filters * 16, stride=2),
            SplitIdentityPointwise(filters * 16),

            SplitIdentityBlock(filters * 16, True, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False),

            blocks.DepthwiseConv2d(filters * 16, filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 24),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(filters * 24, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class IdentityDWBlock(nn.Module):
    def __init__(
        self,
        inp: int,
        combined: bool = True,
        combine: bool = True
    ):
        super().__init__()

        self.split = blocks.ChannelChunk(2)
        self.branch1 = nn.Identity()
        self.branch2 = blocks.DepthwiseConv2d(inp, inp)
        self.combine1 = blocks.Combine('CONCAT')

        self.pointwise = blocks.PointwiseBlock(inp * 2, inp // 2)
        self.combine2 = blocks.Combine('CONCAT')

    def forward(self, x):
        _, x2 = self.split(x)
        out = self.combine1([self.branch1(x), self.branch2(x)])
        out = self.combine2([self.branch1(x2), self.pointwise(out)])
        return out


def mobilenet_g7(pretrained: bool = False, pth: str = None):
    model = MobileNetG7()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetG7(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 16):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters * 1, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            SplitIdentityPointwiseX2(filters),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            SplitIdentityPointwiseX2(filters * 2),

            IdentityDWBlock(filters * 4, False),

            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4),

            IdentityDWBlock(filters * 8, False),

            blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8),

            IdentityDWBlock(filters * 16, False, False),
            IdentityDWBlock(filters * 16, False, False),
            # IdentityDWBlock(filters * 16, False, False),
            IdentityDWBlock(filters * 16, False, False),
            IdentityDWBlock(filters * 16, False),

            blocks.GaussianFilter(filters * 16, stride=2),
            SplitIdentityPointwise(filters * 16),

            # IdentityDWBlock(filters * 16, True, False),
            IdentityDWBlock(filters * 16, False, False),
            # IdentityDWBlock(filters * 16, False),

            blocks.DepthwiseConv2d(filters * 16, filters * 16),
            blocks.PointwiseBlock(filters * 16, 496),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Linear(496, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_g12356(pretrained: bool = False, pth: str = None):
    model = MobileNetG12356()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetG12356(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters * 1, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            SplitIdentityPointwiseX2(filters),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            SplitIdentityPointwiseX2(filters * 2, False),

            SplitIdentityBlockFilters32(filters * 4, False),

            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4, False),

            SplitIdentityBlockFilters32(filters * 8, False),

            blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8, False),

            SplitIdentityBlockFilters32(filters * 16, False, False),
            SplitIdentityBlockFilters32(filters * 16, False, False),
            SplitIdentityBlockFilters32(filters * 16, False, False),
            SplitIdentityBlockFilters32(filters * 16, False, False),
            SplitIdentityBlockFilters32(filters * 16, False),

            blocks.GaussianFilter(filters * 16, stride=2),
            SplitIdentityPointwiseX2(filters * 16),

            blocks.SharedDepthwiseConv2d(filters * 32, t=8),
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


def mobilenet_g6(pretrained: bool = False, pth: str = None):
    model = MobileNetG6()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MobileNetG6(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters * 1, stride=2),

            Filters32(filters),
            blocks.PointwiseBlock(filters, filters * 2),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            blocks.PointwiseBlock(filters * 2, filters * 4),

            Filters32(filters * 4),
            blocks.PointwiseBlock(filters * 4, filters * 4),

            blocks.GaussianFilter(filters * 4, stride=2),
            blocks.PointwiseBlock(filters * 4, filters * 8),

            Filters32(filters * 8),
            blocks.PointwiseBlock(filters * 8, filters * 8),

            blocks.GaussianFilter(filters * 8, stride=2),
            blocks.PointwiseBlock(filters * 8, filters * 16),

            Filters32(filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            Filters32(filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            Filters32(filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            Filters32(filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            Filters32(filters * 16),
            blocks.PointwiseBlock(filters * 16, filters * 16),

            blocks.GaussianFilter(filters * 16, stride=2),
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


def micronet_a1_5(pretrained: bool = False, pth: str = None):
    model = MicroNetA15()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MicroNetA15(nn.Module):
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

            blocks.DepthwiseConv2d(filters * 4, filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4, False),

            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False),

            blocks.DepthwiseConv2d(filters * 8, filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8, False),

            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False),

            blocks.DepthwiseConv2d(filters * 16, filters * 16, stride=2),
            SplitIdentityPointwise(filters * 16),

            SplitIdentityBlock(filters * 16, True, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False),

            blocks.SharedDepthwiseConv2d(filters * 16, t=8),
            blocks.PointwiseBlock(filters * 16, 512),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def micronet_b1_5(pretrained: bool = False, pth: str = None):
    model = MicroNetB15()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MicroNetB15(nn.Module):
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

            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4, False),

            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False),

            blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8, False),

            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False),

            blocks.GaussianFilter(filters * 16, stride=2),
            SplitIdentityPointwise(filters * 16),

            SplitIdentityBlock(filters * 16, True, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False),

            blocks.SharedDepthwiseConv2d(filters * 16, t=8),
            blocks.PointwiseBlock(filters * 16, 512),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def micronet_c1_5(pretrained: bool = False, pth: str = None):
    model = MicroNetC15()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MicroNetC15(nn.Module):
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

            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4, False),

            SplitIdentityBlockFilter(filters * 8, False, False),
            SplitIdentityBlockFilter(filters * 8, False, False),
            SplitIdentityBlockFilter(filters * 8, False),

            blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8, False),

            SplitIdentityBlockFilter(filters * 16, False, False),
            SplitIdentityBlockFilter(filters * 16, False, False),
            SplitIdentityBlockFilter(filters * 16, False, False),
            SplitIdentityBlockFilter(filters * 16, False, False),
            SplitIdentityBlockFilter(filters * 16, False),

            blocks.GaussianFilter(filters * 16, stride=2),
            SplitIdentityPointwise(filters * 16),

            SplitIdentityBlockFilter(filters * 16, True, False),
            SplitIdentityBlockFilter(filters * 16, False, False),
            SplitIdentityBlockFilter(filters * 16, False),

            blocks.SharedDepthwiseConv2d(filters * 16, t=8),
            blocks.PointwiseBlock(filters * 16, 512),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def micronet_a1_0(pretrained: bool = False, pth: str = None):
    model = MicroNetA10(3, 1000, 32)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MicroNetA10(nn.Module):
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

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            SplitIdentityPointwiseX2(filters * 2, False),

            SplitIdentityBlock(filters * 4, False),

            blocks.DepthwiseConv2d(filters * 4, filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4, False),

            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False),

            blocks.DepthwiseConv2d(filters * 8, filters * 8, stride=2),
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


def micronet_b1_0(pretrained: bool = False, pth: str = None):
    model = MicroNetB10(3, 1000, 32)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MicroNetB10(nn.Module):
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

            blocks.GaussianFilter(filters * 2, stride=2),
            SplitIdentityPointwiseX2(filters * 2, False),

            SplitIdentityBlock(filters * 4, False),

            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4, False),

            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False),

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


def micronet_c1_0(pretrained: bool = False, pth: str = None):
    model = MicroNetC10(3, 1000, 32)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MicroNetC10(nn.Module):
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


def micronet_b2_0(pretrained: bool = False, pth: str = None):
    model = MicroNetB20()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MicroNetB20(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            blocks.PointwiseBlock(filters, filters * 2),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            SplitIdentityPointwiseX2(filters * 2, False),

            SplitIdentityBlock(filters * 4, False),

            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4, False),

            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False),

            blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8, False),

            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False),

            blocks.GaussianFilter(filters * 16, stride=2),
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


def micronet_c2_0(pretrained: bool = False, pth: str = None):
    model = MicroNetC20()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MicroNetC20(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            blocks.PointwiseBlock(filters, filters * 2),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            SplitIdentityPointwiseX2(filters * 2, False),

            SplitIdentityBlockFilters32(filters * 4, False),

            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4, False),

            SplitIdentityBlockFilters32(filters * 8, False, False),
            SplitIdentityBlockFilters32(filters * 8, False, False),
            SplitIdentityBlockFilters32(filters * 8, False),

            blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8, False),

            SplitIdentityBlockFilters32(filters * 16, False, False),
            SplitIdentityBlockFilters32(filters * 16, False, False),
            SplitIdentityBlockFilters32(filters * 16, False, False),
            SplitIdentityBlockFilters32(filters * 16, False),

            blocks.GaussianFilter(filters * 16, stride=2),
            SplitIdentityPointwise(filters * 16),

            SplitIdentityBlockFilters32(filters * 16, True, False),
            SplitIdentityBlockFilters32(filters * 16, False, False),
            SplitIdentityBlockFilters32(filters * 16, False),

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


def micronet_b2_5(pretrained: bool = False, pth: str = None):
    model = MicroNetB25()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MicroNetB25(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            blocks.PointwiseBlock(filters, filters * 2),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            SplitIdentityPointwiseX2(filters * 2, False),

            SplitIdentityBlock(filters * 4, False),

            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4, False),

            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False),

            blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8, False),

            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False),

            blocks.GaussianFilter(filters * 16, stride=2),
            SplitIdentityPointwise(filters * 16),

            SplitIdentityBlock(filters * 16, True, False),
            SplitIdentityBlock(filters * 16, False, False),
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


def micronet_d2_0(pretrained: bool = False, pth: str = None):
    model = MicroNetD20()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MicroNetD20(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 16):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            blocks.PointwiseBlock(filters, filters * 2),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            SplitIdentityPointwiseX2(filters * 2, False),

            SplitIdentityBlock(filters * 4, False),

            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4, False),

            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False),

            blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8, False),

            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            # SplitIdentityBlock(filters * 16, False, False),
            # SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False),

            blocks.GaussianFilter(filters * 16, stride=2),
            SplitIdentityPointwiseX2(filters * 16),

            SplitIdentityBlock(filters * 32, True, False),
            SplitIdentityBlock(filters * 32, False, False),
            SplitIdentityBlock(filters * 32, False, False),
            SplitIdentityBlock(filters * 32, False, False),
            SplitIdentityBlock(filters * 32, False, False),
            SplitIdentityBlock(filters * 32, False),

            blocks.SharedDepthwiseConv2d(filters * 32, t=8),
            blocks.PointwiseBlock(filters * 32, 512),
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


def micronet_b5_0(pretrained: bool = False, pth: str = None):
    model = MicroNetB50()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MicroNetB50(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 24):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            blocks.PointwiseBlock(filters, filters * 2),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            SplitIdentityPointwiseX2(filters * 2, False),

            SplitIdentityBlock(filters * 4, False, False),
            SplitIdentityBlock(filters * 4, False, False),
            SplitIdentityBlock(filters * 4, False),

            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4, False),

            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False, False),
            SplitIdentityBlock(filters * 8, False),

            blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8, False),

            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False, False),
            SplitIdentityBlock(filters * 16, False),

            blocks.GaussianFilter(filters * 16, stride=2),
            SplitIdentityPointwiseX2(filters * 16),

            SplitIdentityBlock(filters * 32, True, False),
            SplitIdentityBlock(filters * 32, False, False),
            SplitIdentityBlock(filters * 32, False, False),
            SplitIdentityBlock(filters * 32, False, False),
            SplitIdentityBlock(filters * 32, False, False),
            SplitIdentityBlock(filters * 32, False, False),
            SplitIdentityBlock(filters * 32, False, False),
            SplitIdentityBlock(filters * 32, False, False),
            SplitIdentityBlock(filters * 32, False),

            blocks.SharedDepthwiseConv2d(filters * 32, t=8),
            blocks.PointwiseBlock(filters * 32, 512),
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


def micronet_c2_5(pretrained: bool = False, pth: str = None):
    model = MicroNetC25()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MicroNetC25(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=2),

            blocks.DepthwiseConv2d(filters, filters),
            blocks.PointwiseBlock(filters, filters * 2),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=2),
            SplitIdentityPointwiseX2(filters * 2, False),

            SplitIdentityBlockFilters32(filters * 4, False),

            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4, False),

            SplitIdentityBlockFilters32(filters * 8, False, False),
            SplitIdentityBlockFilters32(filters * 8, False, False),
            SplitIdentityBlockFilters32(filters * 8, False, False),
            SplitIdentityBlockFilters32(filters * 8, False),

            blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8, False),

            SplitIdentityBlockFilters32(filters * 16, False, False),
            SplitIdentityBlockFilters32(filters * 16, False, False),
            SplitIdentityBlockFilters32(filters * 16, False, False),
            SplitIdentityBlockFilters32(filters * 16, False, False),
            SplitIdentityBlockFilters32(filters * 16, False, False),
            SplitIdentityBlockFilters32(filters * 16, False),

            blocks.GaussianFilter(filters * 16, stride=2),
            SplitIdentityPointwise(filters * 16),

            SplitIdentityBlockFilters32(filters * 16, True, False),
            SplitIdentityBlockFilters32(filters * 16, False, False),
            SplitIdentityBlockFilters32(filters * 16, False, False),
            SplitIdentityBlockFilters32(filters * 16, False),

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
