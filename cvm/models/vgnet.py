from functools import partial
import os
import torch
import torch.nn as nn
from .core import blocks, export, config
from typing import Any, List, OrderedDict


class HalfIdentityBlock(nn.Module):
    def __init__(
        self,
        inp: int,
        g: int = 1,
        se_ratio: float = 0.0,
        kernels: str = 'random'
    ):
        super().__init__()

        if kernels == 'random':
            self.half3x3 = blocks.Conv2d3x3(
                inp // 2, inp // 2, groups=(inp // 2) // min(inp // 2, g))
        elif kernels == 'edge':
            self.half3x3 = nn.Sequential(
                blocks.EdgeDetection(inp // 2),
                nn.BatchNorm2d(inp // 2)
            )
        elif kernels == 'edge_gassian':
            self.half3x3 = nn.Sequential(
                blocks.FixedConv2d(inp // 2),
                nn.BatchNorm2d(inp // 2)
            )
        else:
            ValueError(f'')

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
        kernels: str = 'random',
        thumbnail: bool = False,
        **kwargs: Any
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 2
        strides = [FRONT_S, 2, 2, 2]

        self.features = nn.Sequential(OrderedDict([
            ('stem', blocks.Conv2dBlock(
                in_channels, channels[0], stride=FRONT_S
            ))
        ]))

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
                    se_ratio,
                    kernels
                )
            )

        if kernels == 'random':
            self.features.stage4.append(nn.Sequential(
                # blocks.DepthwiseConv2d(channels[-1], channels[-1]),
                blocks.SharedDepthwiseConv2d(channels[-1], t=8),
                blocks.PointwiseBlock(channels[-1], channels[-1]),
            ))
        elif kernels == 'edge_gassian':
            self.features.stage4.append(nn.Sequential(
                blocks.FixedConv2d(channels[-1]),
                nn.BatchNorm2d(channels[-1]),
                blocks.PointwiseBlock(channels[-1], channels[-1]),
            ))
        elif kernels == 'edge':
            self.features.stage4.append(nn.Sequential(
                blocks.EdgeDetection(channels[-1]),
                nn.BatchNorm2d(channels[-1]),
                blocks.PointwiseBlock(channels[-1], channels[-1]),
            ))

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels[-1], num_classes)

    def make_layers(self, inp, oup, s, m, n, g, se_ratio, fixed):
        layers = [
            DownsamplingBlock(inp, oup, stride=s, method=m, se_ratio=se_ratio)
        ]
        for _ in range(n - 1):
            layers.append(HalfIdentityBlock(oup, g, se_ratio, fixed))

        layers.append(blocks.Combine('CONCAT'))
        return blocks.Stage(layers)

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
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.2-vgnets-weights/vgnetg_1_0mp-533cb12b.pth')
def vgnetg_1_0mp(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [28, 56, 112, 224, 368]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [4, 7, 13, 2]
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
@blocks.nonlinear(partial(nn.SiLU, inplace=True))
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.2-vgnets-weights/vgnetg_1_0mp_silu-e56f1ed9.pth')
def vgnetg_1_0mp_silu(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [28, 56, 112, 224, 368]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [4, 7, 13, 2]
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.2-vgnets-weights/vgnetg_1_0mp_se-4ddbde88.pth')
def vgnetg_1_0mp_se(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [28, 56, 112, 224, 368]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [4, 7, 13, 2]
    kwargs['se_ratio'] = 0.25
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.2-vgnets-weights/vgnetg_1_5mp-cbaa0f5d.pth')
def vgnetg_1_5mp(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 64, 128, 256, 512]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [3, 7, 14, 2]
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.2-vgnets-weights/vgnetc_1_5mp-3711bf00.pth')
def vgnetc_1_5mp(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 64, 128, 256, 512]
    kwargs['downsamplings'] = ['dwconv', 'dwconv', 'dwconv', 'dwconv']
    kwargs['layers'] = [3, 7, 14, 2]
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.2-vgnets-weights/vgnetf_1_5mp-21d7e648.pth')
def vgnetf_1_5mp(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 64, 128, 256, 512]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [3, 7, 14, 2]
    kwargs['kernels'] = 'edge_gassian'
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.2-vgnets-weights/vgnete_1_5mp-8416471a.pth')
def vgnete_1_5mp(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 64, 128, 256, 512]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [3, 7, 14, 2]
    kwargs['kernels'] = 'edge'
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.2-vgnets-weights/vgnetg_1_5mp_silu-0edf5431.pth')
@blocks.nonlinear(partial(nn.SiLU, inplace=True))
def vgnetg_1_5mp_silu(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 64, 128, 256, 512]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [3, 7, 14, 2]
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.2-vgnets-weights/vgnetg_1_5mp_se-a759664f.pth')
def vgnetg_1_5mp_se(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 64, 128, 256, 512]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [3, 7, 14, 2]
    kwargs['se_ratio'] = 0.25
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.2-vgnets-weights/vgnetg_2_0mp-aa9fa383.pth')
def vgnetg_2_0mp(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 72, 168, 376, 512]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [3, 6, 13, 2]
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.2-vgnets-weights/vgnetg_2_0mp_se-bcf70864.pth')
def vgnetg_2_0mp_se(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 72, 168, 376, 512]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [3, 6, 13, 2]
    kwargs['se_ratio'] = 0.25
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.2-vgnets-weights/vgnetg_2_5mp-378e2472.pth')
def vgnetg_2_5mp(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 80, 192, 400, 544]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [3, 6, 16, 2]
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.2-vgnets-weights/vgnetg_2_5mp_se-8c373f00.pth')
def vgnetg_2_5mp_se(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 80, 192, 400, 544]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [3, 6, 16, 2]
    kwargs['se_ratio'] = 0.25
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
def vgnetg_5_0mp(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 88, 216, 456, 856]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [4, 7, 15, 5]
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
def vgnetg_5_0mp_se(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 88, 216, 456, 856]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [4, 7, 15, 5]
    kwargs['se_ratio'] = 0.25
    return _vgnet(pretrained, pth, progress, **kwargs)
