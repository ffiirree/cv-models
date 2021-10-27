from functools import partial
import os
import torch
import torch.nn as nn
from .core import blocks, Conv2dBlock, Conv2d1x1Block, export
from typing import Any


class Residual(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x):
        return self[0](x) + x


@export
class ConvMixer(nn.Module):
    @blocks.normalizer(position='after')
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        h=None,
        depth=None,
        kernel_size: int = 9,
        patch_size: int = 7,
        **kwargs: Any
    ):
        super().__init__()

        self.features = nn.Sequential(
            Conv2dBlock(in_channels, h, patch_size, stride=patch_size),

            *[nn.Sequential(
                Residual(
                    Conv2dBlock(h, h, kernel_size, groups=h, padding='same')
                ),
                Conv2d1x1Block(h, h)
            ) for _ in range(depth)]
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(h, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _conv_mixer(
    h,
    depth,
    kernel_size: int = 9,
    patch_size: int = 7,
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):

    model = ConvMixer(h=h, depth=depth, kernel_size=kernel_size,
                      patch_size=patch_size, **kwargs)

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
@blocks.nonlinear(nn.GELU)
def conv_mixer_1536_20_k9_p7(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _conv_mixer(1536, 20, 9, 7, pretrained, pth, progress, **kwargs)


@export
@blocks.nonlinear(nn.GELU)
def conv_mixer_1536_20_k3_p7(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _conv_mixer(1536, 20, 3, 7, pretrained, pth, progress, **kwargs)


@export
@blocks.nonlinear(nn.GELU)
def conv_mixer_1024_20_k9_p14(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _conv_mixer(1024, 20, 9, 14, pretrained, pth, progress, **kwargs)


@export
@blocks.nonlinear(nn.GELU)
def conv_mixer_1024_16_k9_p7(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _conv_mixer(1024, 16, 9, 7, pretrained, pth, progress, **kwargs)


@export
@blocks.nonlinear(nn.GELU)
def conv_mixer_1024_12_k8_p7(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _conv_mixer(1024, 12, 8, 7, pretrained, pth, progress, **kwargs)


@export
@blocks.nonlinear(partial(nn.ReLU, inplace=True))
def conv_mixer_768_32_k7_p7(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _conv_mixer(768, 32, 7, 7, pretrained, pth, progress, **kwargs)


@export
@blocks.nonlinear(partial(nn.ReLU, inplace=True))
def conv_mixer_768_32_k3_p14(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _conv_mixer(768, 32, 3, 14, pretrained, pth, progress, **kwargs)


@export
@blocks.nonlinear(nn.GELU)
def conv_mixer_512_16_k8_p7(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _conv_mixer(512, 16, 8, 7, pretrained, pth, progress, **kwargs)


@export
@blocks.nonlinear(nn.GELU)
def conv_mixer_512_12_k8_p7(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _conv_mixer(512, 12, 8, 7, pretrained, pth, progress, **kwargs)
