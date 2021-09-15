import torch
import torch.nn as nn
from .core import blocks
from typing import Any

__all__ = ['Mixer']


class MlpBlock(nn.Sequential):
    def __init__(self, channels):
        super().__init__(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1)
        )


class MixerBlock(nn.Module):
    def __init__(self, C, S):
        super().__init__()

        self.token_mixing = MlpBlock(S)
        self.channel_mixing = MlpBlock(C)

    def forward(self, x):
        identity = x
        # nn.LayerNorm()
        x = torch.transpose(x, 1, 3)
        x = self.token_mixing(x)
        x = torch.transpose(x, 1, 3)
        x = identity + x
        # nn.LayerNorm()
        return x + self.channel_mixing(x)


class Mixer(nn.Module):
    r'''
    See: https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_mixer.py
    '''

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        P: int = 32,
        C: int = 768,
        num_blocks: int = 12
    ):
        super().__init__()

        self.num_blocks = num_blocks

        self.stem = nn.Conv2d(in_channels, C, kernel_size=P, stride=P)
        self.mixer = nn.Sequential(
            *[MixerBlock(C) for _ in self.num_blocks]
        )
        self.avg = nn.AdaptiveAvgPool2d((1))
        self.fc = nn.Linear(C, num_classes)

    def forward(self, x):
        x = self.stem(x)
        # n c h w -> n c 1 s
        x = torch.reshape(x, (x.shape[0], x.shape[1], 1, -1))
        # n c 1 s -> n s 1 c
        # x = x.permute(0, 3, 2, 1)
        x = self.mixer(x)
        x = self.avg(x)
        x = self.fc(x)

        return x
