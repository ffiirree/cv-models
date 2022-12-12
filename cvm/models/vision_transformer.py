import torch
import torch.nn as nn

from .ops import blocks
from .utils import export, config, load_from_local_or_url
from typing import Any
from functools import partial


class MultiheadSelfAttention(nn.MultiheadAttention):
    def forward(self, x):
        x, _ = super().forward(x, x, x, need_weights=False)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads: int = 8,
        qkv_bias: bool = False,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.,
        attn_dropout_rate: float = 0.,
        drop_path_rate: float = 0.,
        normalizer_fn: nn.Module = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        self.msa = nn.Sequential(
            normalizer_fn(embed_dim),
            MultiheadSelfAttention(embed_dim, num_heads, dropout=attn_dropout_rate, bias=qkv_bias, batch_first=True),
            nn.Dropout(dropout_rate),
            blocks.StochasticDepth(1 - drop_path_rate)
        )

        self.mlp = nn.Sequential(
            normalizer_fn(embed_dim),
            blocks.MlpBlock(embed_dim, int(embed_dim * mlp_ratio), dropout_rate=dropout_rate),
            blocks.StochasticDepth(1 - drop_path_rate)
        )

    def forward(self, x):
        x = x + self.msa(x)
        x = x + self.mlp(x)
        return x


@export
class VisionTransformer(nn.Module):
    r"""
    Paper: An Image is Worth 16x16 Words. Transformers for Image Recognition at Scale, https://arxiv.org/abs/2010.11929
    """
    def __init__(
        self,
        image_size: int = 224,
        in_channels: int = 3,
        num_classes: int = 1000,
        patch_size: int = 16,
        hidden_dim: int = 768,
        num_blocks: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        dropout_rate: float = 0.,
        attn_dropout_rate: float = 0.,
        drop_path_rate: float = 0.,
        classifier: str = 'token',
        normalizer_fn: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        **kwargs: Any
    ):
        super().__init__()

        self.num_patches = (image_size // patch_size) ** 2
        self.classifier = classifier

        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.positions = nn.Parameter(torch.normal(mean=0.0, std=0.02, size=[1, self.num_patches + 1, hidden_dim]))

        self.embedding = nn.Conv2d(in_channels, hidden_dim, patch_size, stride=patch_size)

        self.drop = nn.Dropout(dropout_rate)

        # encoder
        self.encoder = nn.Sequential(*[
            EncoderBlock(
                hidden_dim, num_heads, qkv_bias=qkv_bias, mlp_ratio=mlp_ratio,
                dropout_rate=dropout_rate, attn_dropout_rate=attn_dropout_rate,
                drop_path_rate=drop_path_rate, normalizer_fn=normalizer_fn
            ) for _ in range(num_blocks)
        ])

        self.norm = normalizer_fn(hidden_dim)

        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # NCHW -> (N, hidden_dim, NP_H, NP_W)
        x = self.embedding(x)
        # (N, hidden_dim, NP_H, NP_W) -> (N, hidden_dim, NP)
        x = torch.flatten(x, start_dim=2)
        # (N, hidden_dim, NP) -> (N, NP, hidden_dim)
        x = x.permute(0, 2, 1)

        class_tokens = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([class_tokens, x], dim=1) + self.positions

        x = self.drop(x)
        x = self.encoder(x)
        x = self.norm(x)

        x = x[:, 0] if self.classifier == 'token' else x.mean(dim=1)
        return self.head(x)


def _vit(
    image_size: int = 224,
    patch_size: int = 32,
    hidden_dim: int = 768,
    num_blocks: int = 12,
    num_heads: int = 12,
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    model = VisionTransformer(image_size, patch_size=patch_size, hidden_dim=hidden_dim,
                              num_blocks=num_blocks, num_heads=num_heads,
                              normalizer_fn=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.1.1-vit-weights/torch-vit_b_32-f0b6fb13.pth')
def vit_b_32(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vit(224, 32, 768, 12, 12, pretrained, pth, progress, **kwargs)


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.1.1-vit-weights/torch-vit_b_16-1d93d631.pth')
def vit_b_16(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vit(224, 16, 768, 12, 12, pretrained, pth, progress, **kwargs)


@export
def vit_l_32(pretrained: bool = True, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vit(224, 32, 1024, 24, 16, pretrained, pth, progress, **kwargs)


@export
def vit_l_16(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vit(224, 16, 1024, 24, 16, pretrained, pth, progress, **kwargs)


@export
def vit_h_32(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vit(224, 32, 1280, 32, 16, pretrained, pth, progress, **kwargs)


@export
def vit_h_16(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vit(224, 16, 1280, 32, 16, pretrained, pth, progress, **kwargs)
