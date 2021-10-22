import os
import torch
import torch.nn as nn
from .core import blocks
from typing import Any

__all__ = ['micronet_se1_0', 'micronet_se2_0', 'micronet_se2_5']


class SplitIdentityBlock(nn.Module):
    def __init__(
        self,
        inp: int,
        combined: bool = True,
        combine: bool = True
    ):
        super().__init__()

        self.split = blocks.ChannelChunk(2) if combined else nn.Identity()
        self.branch1 = nn.Identity()
        self.branch2 = blocks.DepthwiseConv2d(inp // 2, inp // 2)
        self.combine1 = blocks.Combine('CONCAT')

        self.pointwise = nn.Sequential(
            blocks.PointwiseBlock(inp, inp // 2),
            blocks.SEBlock(inp // 2, 0.25)
        )
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

        self.split = blocks.ChannelChunk(2)
        self.branch1 = nn.Identity()

        self.branch2 = nn.Sequential(
            blocks.PointwiseBlock(inp, inp // 2),
            blocks.SEBlock(inp // 2, 0.25)
        )
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

        self.branch2 = nn.Sequential(
            blocks.PointwiseBlock(inp, inp),
            blocks.SEBlock(inp, 0.25)
        )
        self.combine = blocks.Combine('CONCAT') if combine else nn.Identity()

    def forward(self, x):
        out = self.combine([self.branch1(x), self.branch2(x)])
        return out


def micronet_se1_0(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = MicroNetD10(**kwargs)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MicroNetD10(nn.Module):
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
            blocks.PointwiseBlock(filters, filters * 2),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=FRONT_S),
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

            blocks.SharedDepthwiseConv2d(filters * 16, t=8),
            blocks.PointwiseBlock(filters * 16, 360),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(360, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# def micronet_se1_0(pretrained: bool = False, pth: str = None, **kwargs: Any):
#     model = MicroNetSE10(**kwargs)
#     if pretrained and pth is not None:
#         model.load_state_dict(torch.load(os.path.expanduser(pth)))
#     return model


# class MicroNetSE10(nn.Module):
#     """Params: 1M"""
#     @blocks.batchnorm(position='after')
#     # @blocks.nonlinear(nn.SiLU)
#     def __init__(
#         self,
#         in_channels: int = 3,
#         num_classes: int = 1000,
#         filters: int = 32,
#         thumbnail: bool = False
#     ):
#         super().__init__()

#         FRONT_S = 1 if thumbnail else 2

#         self.features = nn.Sequential(
#             blocks.Conv2dBlock(in_channels, filters, stride=FRONT_S),

#             SplitIdentityBlock(filters),

#             blocks.DepthwiseConv2d(filters, filters, stride=FRONT_S),
#             SplitIdentityPointwiseX2(filters, False),

#             SplitIdentityBlock(filters * 2, False),

#             blocks.GaussianFilter(filters * 2, stride=2),
#             SplitIdentityPointwiseX2(filters * 2, False),

#             SplitIdentityBlock(filters * 4, False, False),
#             SplitIdentityBlock(filters * 4, False),

#             blocks.GaussianFilter(filters * 4, stride=2),
#             SplitIdentityPointwiseX2(filters * 4, False),

#             SplitIdentityBlock(filters * 8, False, False),
#             SplitIdentityBlock(filters * 8, False, False),
#             SplitIdentityBlock(filters * 8, False, False),
#             SplitIdentityBlock(filters * 8, False, False),
#             SplitIdentityBlock(filters * 8, False),

#             blocks.GaussianFilter(filters * 8, stride=2),
#             SplitIdentityPointwiseX2(filters * 8, False),

#             SplitIdentityBlock(filters * 16, False),

#             blocks.DepthwiseBlock(filters * 16, filters * 16),
#             blocks.PointwiseBlock(filters * 16, 380),
#         )

#         self.avg = nn.AdaptiveAvgPool2d((1, 1))
#         self.classifier = nn.Linear(380, num_classes)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.avg(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x


def micronet_se2_0(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = MicroNetSE20(**kwargs)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MicroNetSE20(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        filters: int = 32,
        thumbnail: bool = False
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 2

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=FRONT_S),

            blocks.DepthwiseConv2d(filters, filters),
            blocks.PointwiseBlock(filters, filters * 2),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=FRONT_S),
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


def micronet_se2_5(pretrained: bool = False, pth: str = None, **kwargs: Any):
    model = MicroNetSE25(**kwargs)
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class MicroNetSE25(nn.Module):
    @blocks.batchnorm(position='after')
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        filters: int = 32,
        thumbnail: bool = False
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 2

        self.features = nn.Sequential(
            blocks.Conv2dBlock(in_channels, filters, stride=FRONT_S),

            blocks.DepthwiseConv2d(filters, filters),
            blocks.PointwiseBlock(filters, filters * 2),

            blocks.DepthwiseConv2d(filters * 2, filters * 2, stride=FRONT_S),
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