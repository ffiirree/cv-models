'''
Papers:
    [ViT]  An Image is Worth 16x16 Words. Transformers for Image Recognition at Scale
    [DeiT] Training data-efficient image transformers & distillation through attention
Others:
    https://github.com/google-research/vision_transformer
'''
import os
import torch
import torch.nn as nn
from .core import blocks, export, config
from typing import Any


@export
class VisionTransformer(nn.Module):
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
        dropout_rate: float = 0.,
        attn_dropout_rate: float = 0.,
        drop_path_rate: float = 0.,
        classifier: str = 'token',
        distilled: bool = False,
        **kwargs: Any
    ):
        super().__init__()

        num_patches = (image_size // patch_size) ** 2
        self.classifier = classifier
        self.distilled = distilled

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.dist_token = nn.Parameter(torch.randn(1, 1, hidden_dim)) if distilled else None
        self.positions = nn.Parameter(
            torch.randn(num_patches + (1 if not distilled else 2), hidden_dim))

        self.embedding = nn.Conv2d(
            in_channels, hidden_dim, patch_size, stride=patch_size)

        self.drop = nn.Dropout(dropout_rate)

        # encoder
        self.encoder = nn.Sequential(*[
            blocks.EncoderBlock(
                hidden_dim, num_heads, mlp_ratio=mlp_ratio,
                dropout_rate=dropout_rate, attn_dropout_rate=attn_dropout_rate, drop_path_rate=drop_path_rate
            ) for _ in range(num_blocks)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        self.head = nn.Linear(hidden_dim, num_classes)

        if self.distilled:
            self.head_dist = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        # BCHW -> BNC; N = patches; C = hidden_dim
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)
        if not self.distilled:
            x = torch.cat([cls_tokens, x], dim=1) + self.positions
        else:
            dist_tokens = self.dist_token.repeat(x.shape[0], 1, 1)
            x = torch.cat([cls_tokens, x, dist_tokens], dim=1) + self.positions

        x = self.drop(x)

        x = self.encoder(x)

        x = self.norm(x)

        if not self.distilled:
            x = x[:, 0] if self.classifier == 'token' else x.mean(dim=1)
            return self.head(x)
        else:
            x, x_dist = self.head(x[:, 0]), self.head_dist(x[:, -1])
            if self.training:
                return x, x_dist
            else:
                return (x + x_dist) / 2


def _vit(
    image_size: int = 224,
    patch_size: int = 32,
    hidden_dim: int = 768,
    num_blocks: int = 12,
    num_heads: int = 12,
    distilled: bool = False,
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    model = VisionTransformer(image_size, patch_size=patch_size, hidden_dim=hidden_dim,
                              num_blocks=num_blocks, num_heads=num_heads, distilled=distilled, **kwargs)

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
def vit_b32_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vit(224, 32, 768, 12, 12, False, pretrained, pth, progress, **kwargs)


@export
def vit_b16_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vit(224, 16, 768, 12, 12, False, pretrained, pth, progress, **kwargs)


@export
def vit_l32_224(pretrained: bool = True, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vit(224, 32, 1024, 24, 16, False, pretrained, pth, progress, **kwargs)


@export
def vit_l16_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vit(224, 16, 1024, 24, 16, False, pretrained, pth, progress, **kwargs)


@export
def vit_h32_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vit(224, 32, 1280, 32, 16, False, pretrained, pth, progress, **kwargs)


@export
def vit_h16_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vit(224, 16, 1280, 32, 16, False, pretrained, pth, progress, **kwargs)


@export
def deit_tiny_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vit(224, 16, 192, 12, 3, False, pretrained, pth, progress, **kwargs)


@export
def deit_small_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vit(224, 16, 384, 12, 6, False, pretrained, pth, progress, **kwargs)


@export
def deit_base_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vit(224, 16, 768, 12, 12, False, pretrained, pth, progress, **kwargs)


@export
def deit_tiny_distilled_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vit(224, 16, 192, 12, 3, True, pretrained, pth, progress, **kwargs)


@export
def deit_small_distilled_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vit(224, 16, 384, 12, 6, True, pretrained, pth, progress, **kwargs)


@export
def deit_base_distilled_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _vit(224, 16, 768, 12, 12, True, pretrained, pth, progress, **kwargs)
