from models.core.functional import make_divisible
import os
import torch
import torch.nn as nn
from .core import blocks
from typing import Any


__all__ = ['threepathnet_x1_0', 'threepathnet_x1_5',
           'threepathnet_x2_0', 'threepathnet_x2_5',
           'threepathnet_v2_x1_0']


class ThreePathBlock(nn.Module):
    def __init__(
        self,
        inp: int,
        combined: bool = True,
        combine: bool = True
    ):
        super().__init__()

        self.split = blocks.ChannelChunk(3) if combined else nn.Identity()
        self.branch1 = nn.Identity()
        self.branch2 = blocks.DepthwiseConv2d(inp // 3, inp // 3)
        self.combine1 = blocks.Combine('CONCAT')

        self.pointwise = blocks.PointwiseBlock(inp, inp // 3)
        self.combine2 = blocks.Combine('CONCAT') if combine else nn.Identity()

    def forward(self, x):
        x1, x2, x3 = self.split(x)
        out = self.combine1(
            [self.branch1(x1), self.branch1(x2), self.branch2(x3)])
        out = self.combine2(
            [self.branch1(x1), self.branch1(x3), self.pointwise(out)])
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


class ThreePathDownSampingBlock(nn.Module):
    def __init__(
        self,
        inp: int
    ):
        super().__init__()

        self.depthwise = blocks.DepthwiseConv2d(inp, inp)

        self.pointwise = blocks.PointwiseBlock(inp, make_divisible(inp + 2 * (inp // 3), 6) - inp)
        self.combine = blocks.Combine('CONCAT')

    def forward(self, x):
        x = self.depthwise(x)
        x = self.combine([x, self.pointwise(x)])
        return x


def threepathnet_x1_0(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = ThreePathNet10(**kwargs)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class ThreePathNet10(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        filters: int = 21,
        thumbnail: bool = False
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 2

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=FRONT_S),

            blocks.DepthwiseConv2d(filters, filters),
            SplitIdentityPointwiseX2(filters),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=FRONT_S),
            SplitIdentityPointwiseX2(filters * 2),

            ThreePathBlock(filters * 4),

            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4),

            ThreePathBlock(filters * 8, True, False),
            ThreePathBlock(filters * 8, False),

            blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8),

            ThreePathBlock(filters * 16, True, False),
            ThreePathBlock(filters * 16, False, False),
            ThreePathBlock(filters * 16, False, False),
            ThreePathBlock(filters * 16, False, False),
            ThreePathBlock(filters * 16, False),

            blocks.GaussianFilter(filters * 16, stride=2),
            SplitIdentityPointwise(filters * 16),

            ThreePathBlock(filters * 16),

            blocks.SharedDepthwiseConv2d(filters * 16),
            blocks.PointwiseBlock(filters * 16, 480),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(480, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def threepathnet_x1_5(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = ThreePathNet(**kwargs)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class ThreePathNet(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        filters: int = 21,
        thumbnail: bool = False
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 2

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=FRONT_S),

            blocks.DepthwiseConv2d(filters, filters),
            SplitIdentityPointwiseX2(filters),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=FRONT_S),
            SplitIdentityPointwiseX2(filters * 2),

            ThreePathBlock(filters * 4),

            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4),

            ThreePathBlock(filters * 8, True, False),
            ThreePathBlock(filters * 8, False),

            blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8),

            ThreePathBlock(filters * 16, True, False),
            ThreePathBlock(filters * 16, False, False),
            ThreePathBlock(filters * 16, False, False),
            ThreePathBlock(filters * 16, False),

            blocks.GaussianFilter(filters * 16, stride=2),
            SplitIdentityPointwiseX2(filters * 16),

            ThreePathBlock(filters * 32, True, False),
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


def threepathnet_x2_0(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = ThreePathNetX2_0(**kwargs)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class ThreePathNetX2_0(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        filters: int = 24,
        thumbnail: bool = False
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 2

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=FRONT_S),

            blocks.DepthwiseConv2d(filters, filters),
            SplitIdentityPointwiseX2(filters),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=FRONT_S),
            SplitIdentityPointwiseX2(filters * 2),

            ThreePathBlock(filters * 4),

            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4),

            ThreePathBlock(filters * 8, True, False),
            ThreePathBlock(filters * 8, False, False),
            ThreePathBlock(filters * 8, False),

            blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8),

            ThreePathBlock(filters * 16, True, False),
            ThreePathBlock(filters * 16, False, False),
            ThreePathBlock(filters * 16, False, False),
            ThreePathBlock(filters * 16, False, False),
            ThreePathBlock(filters * 16, False),

            blocks.GaussianFilter(filters * 16, stride=2),
            SplitIdentityPointwiseX2(filters * 16),

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


def threepathnet_x2_5(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = ThreePathNetX2_5(**kwargs)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class ThreePathNetX2_5(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        filters: int = 27,
        thumbnail: bool = False
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 2

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=FRONT_S),

            blocks.DepthwiseConv2d(filters, filters),
            SplitIdentityPointwiseX2(filters),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=FRONT_S),
            SplitIdentityPointwiseX2(filters * 2),

            ThreePathBlock(filters * 4),

            blocks.GaussianFilter(filters * 4, stride=2),
            SplitIdentityPointwiseX2(filters * 4),

            ThreePathBlock(filters * 8, True, False),
            ThreePathBlock(filters * 8, False, False),
            ThreePathBlock(filters * 8, False, False),
            ThreePathBlock(filters * 8, False),

            blocks.GaussianFilter(filters * 8, stride=2),
            SplitIdentityPointwiseX2(filters * 8),

            ThreePathBlock(filters * 16, True, False),
            ThreePathBlock(filters * 16, False, False),
            ThreePathBlock(filters * 16, False, False),
            ThreePathBlock(filters * 16, False, False),
            ThreePathBlock(filters * 16, False, False),
            ThreePathBlock(filters * 16, False, False),
            ThreePathBlock(filters * 16, False),

            blocks.GaussianFilter(filters * 16, stride=2),
            SplitIdentityPointwiseX2(filters * 16),

            ThreePathBlock(filters * 32, True, False),
            ThreePathBlock(filters * 32, False, False),
            ThreePathBlock(filters * 32, False),

            blocks.SharedDepthwiseConv2d(filters * 32),
            blocks.PointwiseBlock(filters * 32, 520),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(520, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def threepathnet_v2_x1_0(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = ThreePathNetV210(**kwargs)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class ThreePathNetV210(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        filters: int = 30,
        thumbnail: bool = False
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 2

        c = [filters]

        for _ in range(5):
            filters = make_divisible(filters + 2 * (filters // 3), 6)
            c.append(filters)

        print(c)

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, c[0], stride=FRONT_S),

            blocks.DepthwiseConv2d(c[0], c[0]),
            ThreePathDownSampingBlock(c[0]),

            blocks.DepthwiseConv2d(c[1], c[1], stride=FRONT_S),
            ThreePathDownSampingBlock(c[1]),

            ThreePathBlock(c[2]),

            blocks.GaussianFilter(c[2], stride=2),
            ThreePathDownSampingBlock(c[2]),

            ThreePathBlock(c[3], True, False),
            ThreePathBlock(c[3], False),

            blocks.GaussianFilter(c[3], stride=2),
            ThreePathDownSampingBlock(c[3]),

            ThreePathBlock(c[4], True, False),
            ThreePathBlock(c[4], False, False),
            ThreePathBlock(c[4], False, False),
            ThreePathBlock(c[4], False, False),
            ThreePathBlock(c[4], False, False),
            ThreePathBlock(c[4], False, False),
            ThreePathBlock(c[4], False),

            blocks.GaussianFilter(c[4], stride=2),
            ThreePathDownSampingBlock(c[4]),

            ThreePathBlock(c[5], True, False),
            ThreePathBlock(c[5], False, False),
            ThreePathBlock(c[5], False),

            blocks.SharedDepthwiseConv2d(c[5]),
            blocks.PointwiseBlock(c[5], 480),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(480, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x