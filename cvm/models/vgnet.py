from functools import partial
import torch
import torch.nn as nn

from .ops import blocks
from .utils import export, config, load_from_local_or_url
from typing import Any, List, OrderedDict


class SharedDepthwiseConv2d(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        t: int = 2,
        bias: bool = False
    ):
        super().__init__()

        self.channels = channels // t
        self.t = t

        if padding is None:
            padding = ((kernel_size - 1) * (dilation - 1) + kernel_size) // 2

        self.mux = blocks.DepthwiseConv2d(self.channels, self.channels, kernel_size,
                                          stride, padding, dilation, bias=bias)

    def forward(self, x):
        x = torch.chunk(x, self.t, dim=1)
        x = [self.mux(xi) for xi in x]
        return torch.cat(x, dim=1)


class HalfIdentityBlock(nn.Module):
    def __init__(
        self,
        inp: int,
        rd_ratio: float = 0.0
    ):
        super().__init__()

        self.half3x3 = blocks.DepthwiseConv2d(inp // 2, inp // 2)
        self.combine = blocks.Combine('CONCAT')
        self.conv1x1 = blocks.PointwiseBlock(inp, inp // 2)

        if rd_ratio > 0.0:
            self.conv1x1 = nn.Sequential(
                blocks.PointwiseBlock(inp, inp // 2),
                blocks.attention_fn(inp // 2, rd_ratio=rd_ratio)
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
        rd_ratio: float = 0.0
    ):
        assert method in ['blur', 'dwconv', 'maxpool'], f'{method}'

        super().__init__()

        if method == 'dwconv' or stride == 1:
            self.downsample = blocks.DepthwiseConv2d(inp, inp, 3, stride)
        elif method == 'maxpool':
            self.downsample = nn.MaxPool2d(kernel_size=3, stride=stride)
        elif method == 'blur':
            self.downsample = blocks.GaussianBlur(inp, stride=stride, sigma_range=(1.0, 1.0), normalize=True)
        else:
            ValueError(f'Unknown downsampling method: {method}.')

        split_chs = 0 if inp > oup else min(oup // 2, inp)

        self.split = None if inp == split_chs else blocks.ChannelSplit([inp - split_chs, split_chs])
        self.conv1x1 = blocks.PointwiseBlock(inp, oup - split_chs)

        if rd_ratio > 0.0:
            self.conv1x1 = nn.Sequential(
                blocks.PointwiseBlock(inp, oup - split_chs),
                blocks.attention_fn(oup - split_chs, rd_ratio=rd_ratio)
            )

        self.halve = nn.Identity()
        if oup > 2 * inp or inp > oup:
            self.halve = nn.Sequential(
                blocks.Combine('CONCAT'),
                blocks.ChannelChunk(2)
            )

    def forward(self, x):
        x = self.downsample(x)
        if self.split is None:
            return self.halve([x, self.conv1x1(x)])
        else:
            _, x2 = self.split(x)
            return self.halve([x2, self.conv1x1(x)])


class VGNet(nn.Module):
    @blocks.normalizer(position='after')
    @blocks.activation(partial(nn.SiLU, inplace=True))
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        channels: List[int] = None,
        downsamplings: List[str] = None,
        layers: List[int] = None,
        rd_ratio: List[float] = [0.0, 0.0, 0.0, 0.0],
        dropout_rate: float = 0.2,
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
                    rd_ratio[i]
                )
            )

        self.features.stage4.append(nn.Sequential(
            blocks.DepthwiseConv2d(channels[-1], channels[-1]),
            # blocks.SharedDepthwiseConv2d(channels[-1], t=8),
            blocks.PointwiseBlock(channels[-1], channels[-1]),
        ))

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate, inplace=True),
            nn.Linear(channels[-1], num_classes)
        )

    def make_layers(self, inp, oup, s, m, n, rd_ratio):
        layers = [
            DownsamplingBlock(inp, oup, stride=s, method=m, rd_ratio=rd_ratio)
        ]
        for _ in range(n - 1):
            layers.append(HalfIdentityBlock(oup, rd_ratio))

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
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
def vgnetc_1_0mp(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [28, 56, 112, 224, 368]
    kwargs['downsamplings'] = ['dwconv', 'dwconv', 'dwconv', 'dwconv']
    kwargs['layers'] = [4, 7, 13, 2]
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.2-vgnets-weights/vgnetg_1_0mp-0f87bf6c.pth')
def vgnetg_1_0mp(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [28, 56, 112, 224, 368]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [4, 7, 13, 2]
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
@blocks.se(partial(nn.SiLU, inplace=True))
@blocks.attention(blocks.SEBlock)
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.2-vgnets-weights/vgnetg_1_0mp_se-914a9c4a.pth')
def vgnetg_1_0mp_se(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [28, 56, 112, 224, 368]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [4, 7, 13, 2]
    kwargs['rd_ratio'] = [0.25, 0.25, 0.25, 0.25]
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.2-vgnets-weights/vgnetg_1_5mp-1ea464de.pth')
def vgnetg_1_5mp(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 64, 128, 256, 512]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [3, 7, 14, 2]
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
@blocks.se(partial(nn.SiLU, inplace=True))
@blocks.attention(blocks.SEBlock)
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.2-vgnets-weights/vgnetg_1_5mp_se-6d9ebf3b.pth')
def vgnetg_1_5mp_se(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 64, 128, 256, 512]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [3, 7, 14, 2]
    kwargs['rd_ratio'] = [0.25, 0.25, 0.25, 0.25]
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.2-vgnets-weights/vgnetg_2_0mp-4594e276.pth')
def vgnetg_2_0mp(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 72, 168, 376, 512]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [3, 6, 13, 2]
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
@blocks.se(partial(nn.SiLU, inplace=True))
@blocks.attention(blocks.SEBlock)
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.2-vgnets-weights/vgnetg_2_0mp_se-132bc3af.pth')
def vgnetg_2_0mp_se(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 72, 168, 376, 512]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [3, 6, 13, 2]
    kwargs['rd_ratio'] = [0.25, 0.25, 0.25, 0.25]
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.2-vgnets-weights/vgnetg_2_5mp-d38ca7ae.pth')
def vgnetg_2_5mp(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 80, 192, 400, 544]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [3, 6, 16, 2]
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
@blocks.se(partial(nn.SiLU, inplace=True))
@blocks.attention(blocks.SEBlock)
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.0.2-vgnets-weights/vgnetg_2_5mp_se-ed87bdb1.pth')
def vgnetg_2_5mp_se(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 80, 192, 400, 544]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [3, 6, 16, 2]
    kwargs['rd_ratio'] = [0.25, 0.25, 0.25, 0.25]
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
def vgnetg_5_0mp(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 88, 216, 456, 856]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [4, 7, 15, 5]
    return _vgnet(pretrained, pth, progress, **kwargs)


@export
@blocks.se(partial(nn.SiLU, inplace=True))
@blocks.attention(blocks.SEBlock)
def vgnetg_5_0mp_se(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['channels'] = [32, 88, 216, 456, 856]
    kwargs['downsamplings'] = ['blur', 'blur', 'blur', 'blur']
    kwargs['layers'] = [4, 7, 15, 5]
    kwargs['rd_ratio'] = [0.25, 0.25, 0.25, 0.25]
    return _vgnet(pretrained, pth, progress, **kwargs)
