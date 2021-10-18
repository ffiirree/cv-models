import os
import torch
import torch.nn as nn
from .core import blocks
from typing import Any

__all__ = ['Mixer', 'mixer_s32_224', 'mixer_s16_224', 'mixer_b32_224',
           'mixer_b16_224', 'mixer_l32_224', 'mixer_l16_224', 'mixer_h14_224']

model_urls = {
    'mixer_s32_224': None,
    'mixer_s16_224': None,
    'mixer_b32_224': None,
    'mixer_b16_224': None,
    'mixer_l32_224': None,
    'mixer_l16_224': None,
    'mixer_h14_224': None
}


class MlpBlock(nn.Sequential):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation_fn: nn.Module = nn.GELU
    ):
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        super().__init__(
            nn.Linear(in_features, hidden_features),
            activation_fn(),
            nn.Linear(hidden_features, out_features)
        )


class MixerBlock(nn.Module):
    def __init__(self, hidden_dim, sequence_len, ratio=(0.5, 4.0)):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.token_mixing = MlpBlock(sequence_len, int(hidden_dim * ratio[0]))
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.channel_mixing = MlpBlock(hidden_dim, int(hidden_dim * ratio[1]))

    def forward(self, x):
        x = x + \
            self.token_mixing(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.channel_mixing(self.norm2(x))
        return x


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
        **kwargs: Any
    ):
        super().__init__()

        self.num_blocks = num_blocks
        self.num_patches = (image_size // patch_size) ** 2

        self.stem = nn.Conv2d(in_channels, hidden_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.mixer = nn.Sequential(
            *[MixerBlock(hidden_dim, self.num_patches) for _ in range(self.num_blocks)]
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
    arch: str,
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
        if pth is not None:
            state_dict = torch.load(os.path.expanduser(pth))
        else:
            state_dict = torch.hub.load_state_dict_from_url(
                model_urls[arch],
                progress=progress
            )
        model.load_state_dict(state_dict)
    return model


def mixer_s32_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mixer('mixer_s32_224', 224, 32, 512, 8, pretrained, pth, progress, **kwargs)


def mixer_s16_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mixer('mixer_s32_224', 224, 16, 512, 8, pretrained, pth, progress, **kwargs)


def mixer_b32_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mixer('mixer_s32_224', 224, 32, 768, 12, pretrained, pth, progress, **kwargs)


def mixer_b16_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mixer('mixer_s32_224', 224, 16, 768, 12, pretrained, pth, progress, **kwargs)


def mixer_l32_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mixer('mixer_s32_224', 224, 32, 1024, 24, pretrained, pth, progress, **kwargs)


def mixer_l16_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mixer('mixer_s32_224', 224, 16, 1024, 24, pretrained, pth, progress, **kwargs)


def mixer_h14_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _mixer('mixer_s32_224', 224, 14, 1280, 32, pretrained, pth, progress, **kwargs)
