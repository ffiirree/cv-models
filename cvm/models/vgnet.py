import os
import torch
import torch.nn as nn
from .core import blocks, export, config
from typing import Any, List


class HalfIdentityBlock(nn.Module):
    def __init__(
        self,
        inp: int,
        g: int = 1,
        se_ratio: float = 0.0
    ):
        super().__init__()

        self.half3x3 = blocks.Conv2d3x3(
            inp // 2, inp // 2, groups=(inp // 2) // min(inp // 2, g))
        self.combine = blocks.Combine('CONCAT')

        self.conv1x1 = blocks.PointwiseBlock(inp, inp // 2)
        if se_ratio > 0.0:
            self.conv1x1 = nn.Sequential(
                blocks.PointwiseBlock(inp, inp // 2),
                blocks.SEBlock(inp // 2, se_ratio)
            )

    def forward(self, x):
        out = self.combine([x[0], self.half3x3(x[1])])
        return [x[1], self.conv1x1(out)]


class DownsamplingBlock(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        stride: int = 2,
        method: str = 'blur',
        se_ratio: float = 0.0
    ):
        assert method in ['blur', 'dwconv', 'maxpool'], f'{method}'

        super().__init__()

        if method == 'dwconv' or stride == 1:
            self.downsample = blocks.DepthwiseConv2d(inp, inp, 3, stride)
        elif method == 'maxpool':
            self.downsample = nn.MaxPool2d(kernel_size=3, stride=stride)
        elif method == 'blur':
            self.downsample = blocks.GaussianFilter(inp, stride=stride)
        else:
            ValueError(f'Unknown downsampling method: {method}.')

        split_chs = 0 if inp > oup else min(oup // 2, inp)

        self.split = blocks.ChannelSplit([inp - split_chs, split_chs])
        self.conv1x1 = blocks.PointwiseBlock(inp, oup - split_chs)

        if se_ratio > 0.0:
            self.conv1x1 = nn.Sequential(
                blocks.PointwiseBlock(inp, oup - split_chs),
                blocks.SEBlock(oup - split_chs, se_ratio)
            )

        self.halve = nn.Identity()
        if oup > 2 * inp or inp > oup:
            self.halve = nn.Sequential(
                blocks.Combine('CONCAT'),
                blocks.ChannelChunk(2)
            )

    def forward(self, x):
        x = self.downsample(x)
        _, x2 = self.split(x)
        return self.halve([x2, self.conv1x1(x)])


class VGNet(nn.Module):
    @blocks.normalizer(position='after')
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        channels: List[int] = None,
        downsamplings: List[str] = None,
        layers: List[int] = None,
        group_widths: List[int] = [1, 1, 1, 1],
        se_ratio: float = 0.0,
        thumbnail: bool = False,
        **kwargs: Any
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 2
        strides = [FRONT_S, 2, 2, 2]

        self.features = nn.Sequential()

        self.features.add_module(
            'stem',
            blocks.Conv2dBlock(in_channels, channels[0], stride=FRONT_S)
        )

        for i in range(len(strides)):
            self.features.add_module(
                f'stage{i+1}',
                self.make_layers(
                    channels[i],
                    channels[i+1],
                    strides[i],
                    downsamplings[i],
                    layers[i],
                    group_widths[i],
                    se_ratio
                )
            )

        self.features.add_module('last', nn.Sequential(
            blocks.SharedDepthwiseConv2d(channels[-1], t=8),
            blocks.PointwiseBlock(channels[-1], channels[-1]),
        ))

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels[-1], num_classes)

    def make_layers(self, inp, oup, s, m, n, g, se_ratio):
        layers = [
            DownsamplingBlock(inp, oup, stride=s, method=m, se_ratio=se_ratio)
        ]
        for _ in range(n - 1):
            layers.append(HalfIdentityBlock(oup, g, se_ratio))

        layers.append(blocks.Combine('CONCAT'))
        return blocks.Stage(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _vgnet(
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    model = VGNet(**kwargs)

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
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.2-vgnets-weights/vgnet_g_1_0mp-baec6e1c.pth')
def vgnet_g_1_0mp(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [28, 56, 112, 224, 368]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [4, 7, 13, 2]
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.2-vgnets-weights/vgnet_g_1_0mp_se-1b12c66e.pth')
def vgnet_g_1_0mp_se(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [28, 56, 112, 224, 368]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [4, 7, 13, 2]
    kwargs['se_ratio'] = 0.25
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.2-vgnets-weights/vgnet_g_1_5mp-1eab7052.pth')
def vgnet_g_1_5mp(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 64, 128, 256, 512]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [3, 7, 14, 2]
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.2-vgnets-weights/vgnet_g_1_5mp_se-d8fc4b39.pth')
def vgnet_g_1_5mp_se(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 64, 128, 256, 512]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [3, 7, 14, 2]
    kwargs['se_ratio'] = 0.25
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
def vgnet_g_2_0mp(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 72, 168, 376, 512]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [3, 6, 13, 2]
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
def vgnet_g_2_0mp_se(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 72, 168, 376, 512]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [3, 6, 13, 2]
    kwargs['se_ratio'] = 0.25
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
def vgnet_g_2_5mp(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 72, 184, 392, 512]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [3, 7, 16, 3]
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
def vgnet_g_2_5mp_se(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 72, 184, 392, 512]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [3, 7, 16, 3]
    kwargs['se_ratio'] = 0.25
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
def vgnet_g_5_0mp(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 88, 216, 456, 856]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [4, 7, 15, 5]
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
def vgnet_g_5_0mp_se(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 88, 216, 456, 856]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [4, 7, 15, 5]
    kwargs['se_ratio'] = 0.25
    return _vgnet(pretrained, pth, progress, **kwargs)
