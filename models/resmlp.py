import os
import torch
import torch.nn as nn
from .core import blocks, export
from typing import Any


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        return self.alpha * x + self.beta


class ResMlpBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        sequence_len,
        layerscale_init: float = 1e-4,
        dropout_rate: float = 0.,
        drop_path_rate: float = 0.
    ):
        super().__init__()

        self.affine_1 = Affine(hidden_dim)
        self.linear_patches = nn.Linear(sequence_len, sequence_len)
        self.layerscale_1 = nn.Parameter(layerscale_init * torch.ones(hidden_dim))
        self.drop1 = blocks.DropPath(1.0 - drop_path_rate)

        self.affine_2 = Affine(hidden_dim)
        self.mlp_channels = blocks.MlpBlock(hidden_dim, hidden_dim * 4, dropout_rate=dropout_rate)
        self.layerscale_2 = nn.Parameter(layerscale_init * torch.ones(hidden_dim))
        self.drop2 = blocks.DropPath(1.0 - drop_path_rate)

    def forward(self, x):
        x = x + self.drop1(self.layerscale_1 * self.linear_patches(self.affine_1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop2(self.layerscale_2 * self.mlp_channels(self.affine_2(x)))
        return x


@export
class ResMLP(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        in_channels: int = 3,
        num_classes: int = 1000,
        patch_size: int = 32,
        hidden_dim: int = 768,
        depth: int = 12,
        dropout_rate: float = 0.,
        drop_path_rate: float = 0.,
        **kwargs: Any
    ):
        super().__init__()

        num_patches = (image_size // patch_size) ** 2

        self.stem = nn.Conv2d(in_channels, hidden_dim,
                              kernel_size=patch_size, stride=patch_size)

        self.blocks = nn.Sequential(
            *[ResMlpBlock(
                hidden_dim,
                num_patches,
                dropout_rate=dropout_rate,
                drop_path_rate=drop_path_rate
            ) for _ in range(depth)]
        )

        self.affine = Affine(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.blocks(x)
        x = self.affine(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


def _resmlp(
    image_size: int = 224,
    patch_size: int = 16,
    hidden_dim: int = 768,
    depth: int = 12,
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    model = ResMLP(image_size, patch_size=patch_size,
                   hidden_dim=hidden_dim, depth=depth, **kwargs)

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
    ...


@export
def resmlp_s12_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _resmlp(224, 16, 384, 12, pretrained, pth, progress, **kwargs)


@export
def resmlp_s24_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _resmlp(224, 16, 384, 24, pretrained, pth, progress, **kwargs)


@export
def resmlp_b24_224(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _resmlp(224, 16, 768, 24, pretrained, pth, progress, **kwargs)
