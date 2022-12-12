from functools import partial
import torch
import torch.nn as nn
from .ops import blocks
from .utils import export, config, load_from_local_or_url
from typing import Any


class MixerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        sequence_len,
        ratio=(0.5, 4.0),
        normalizer_fn: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        dropout_rate: float = 0.,
        drop_path_rate: float = 0.
    ):
        super().__init__()

        self.norm1 = normalizer_fn(hidden_dim)
        self.token_mixing = blocks.MlpBlock(sequence_len, int(hidden_dim * ratio[0]), dropout_rate=dropout_rate)
        self.drop1 = blocks.StochasticDepth(1. - drop_path_rate)

        self.norm2 = normalizer_fn(hidden_dim)
        self.channel_mixing = blocks.MlpBlock(hidden_dim, int(hidden_dim * ratio[1]), dropout_rate=dropout_rate)
        self.drop2 = blocks.StochasticDepth(1. - drop_path_rate)

    def forward(self, x):
        x = x + self.drop1(self.token_mixing(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop2(self.channel_mixing(self.norm2(x)))
        return x


@export
class Mixer(nn.Module):
    r'''
    See: https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_mixer.py
    '''

    def __init__(
        self,
        image_size: int = 224,
        in_channels: int = 3,
        num_classes: int = 1000,
        patch_size: int = 32,
        hidden_dim: int = 768,
        num_blocks: int = 12,
        dropout_rate: float = 0.,
        drop_path_rate: float = 0.,
        **kwargs: Any
    ):
        super().__init__()

        self.num_blocks = num_blocks
        self.num_patches = (image_size // patch_size) ** 2

        self.stem = nn.Conv2d(in_channels, hidden_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.mixer = nn.Sequential(
            *[
                MixerBlock(
                    hidden_dim, self.num_patches, dropout_rate=dropout_rate, drop_path_rate=drop_path_rate
                ) for _ in range(self.num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        # n c h w -> n p c
        x = x.flatten(2).transpose(1, 2)
        x = self.mixer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)

        return x


def _mixer(
    image_size: int = 224,
    patch_size: int = 32,
    hidden_dim: int = 768,
    num_blocks: int = 12,
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    model = Mixer(image_size, patch_size=patch_size,
                  hidden_dim=hidden_dim, num_blocks=num_blocks, **kwargs)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
def mixer_s32_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mixer(224, 32, 512, 8, pretrained, pth, progress, **kwargs)


@export
def mixer_s16_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mixer(224, 16, 512, 8, pretrained, pth, progress, **kwargs)


@export
def mixer_b32_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mixer(224, 32, 768, 12, pretrained, pth, progress, **kwargs)


@export
def mixer_b16_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mixer(224, 16, 768, 12, pretrained, pth, progress, **kwargs)


@export
def mixer_l32_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mixer(224, 32, 1024, 24, pretrained, pth, progress, **kwargs)


@export
def mixer_l16_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mixer(224, 16, 1024, 24, pretrained, pth, progress, **kwargs)


@export
def mixer_h14_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mixer(224, 14, 1280, 32, pretrained, pth, progress, **kwargs)
