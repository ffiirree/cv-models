'''
paper:
    [ConvNeXt] A ConvNet for the 2020s(https://arxiv.org/abs/2201.03545)
official code :
    https://github.com/facebookresearch/ConvNeXt/blob/dcb928723662a1289d31190d09d82378b57b810a/models/convnext.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
from .core import blocks, export, config, load_from_local_or_url
from typing import Any, OrderedDict, List


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

    def extra_repr(self):
        return f'{self.normalized_shape}, eps={self.eps}, data_format={self.data_format}'


class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

    def extra_repr(self):
        return ', '.join([str(dim) for dim in self.dims])


class Scale(nn.Module):
    def __init__(
        self,
        dim: int,
        gamma: float = 1e-6
    ):
        super().__init__()

        self.dim = dim
        self.gamma = nn.Parameter(gamma * torch.ones((dim))) if gamma > 0 else None

    def forward(self, x):
        return self.gamma * x if self.gamma is not None else x

    def extra_repr(self):
        return f'{self.dim}'


class ConvNetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        padding: int = 3,
        survival_prob: float = 0.0,
        layer_scale: float = 1e-6
    ):
        super().__init__()
        self.branch1 = nn.Sequential(
            blocks.DepthwiseConv2d(dim, dim, kernel_size, padding=padding, bias=True),
            Permute([0, 2, 3, 1]),
            LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            Scale(dim, layer_scale),
            Permute([0, 3, 1, 2])
        )

        if survival_prob > 0:
            self.branch1.add_module(str(len(self.branch1)), blocks.DropPath(survival_prob))

        self.branch2 = nn.Identity()
        self.combine = blocks.Combine('ADD')

    def forward(self, x):
        return self.combine([self.branch1(x), self.branch2(x)])


class DownsamplingBlock(nn.Sequential):
    def __init__(
        self,
        inp,
        oup
    ):
        super().__init__(
            LayerNorm(inp, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(inp, oup, 2, stride=2)
        )


@export
class ConvNeXt(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        layers: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.2,
        layer_scale: float = 1e-6,
        thumbnail: bool = False,
        **kwargs: Any
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 4

        self.features = nn.Sequential(OrderedDict([
            ('stem', blocks.Stage(
                nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=FRONT_S),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            ))
        ]))

        survival_probs = [1 - x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]
        for i in range(len(layers)):
            stage = blocks.Stage([
                ConvNetBlock(dims[i], survival_prob=survival_probs[sum(layers[:i]) + j], layer_scale=layer_scale)
                for j in range(layers[i])]
            )
            if i < 3:
                stage.append(DownsamplingBlock(dims[i], dims[i+1]))

            self.features.add_module(f'stage{i + 1}', stage)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.LayerNorm(dims[-1], eps=1e-6),
            nn.Linear(dims[-1], num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


@export
def convnext_tiny(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    model = ConvNeXt(layers=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
def convnext_small(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    model = ConvNeXt(layers=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
def convnext_base(pretrained: bool = False, in_22k=False, pth: str = None, progress: bool = True, **kwargs: Any):
    model = ConvNeXt(layers=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
def convnext_large(pretrained: bool = False, in_22k=False, pth: str = None, progress: bool = True, **kwargs: Any):
    model = ConvNeXt(layers=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
def convnext_xlarge(pretrained: bool = False, in_22k=False, pth: str = None, progress: bool = True, **kwargs: Any):
    model = ConvNeXt(layers=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model
