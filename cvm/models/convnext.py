'''
paper:
    [ConvNeXt] A ConvNet for the 2020s(https://arxiv.org/abs/2201.03545)
official code :
    https://github.com/facebookresearch/ConvNeXt/blob/dcb928723662a1289d31190d09d82378b57b810a/models/convnext.py
'''
import torch
import torch.nn as nn
from .ops import blocks
from .utils import export, config, load_from_local_or_url
from typing import Any, OrderedDict, List


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
            blocks.Permute([0, 2, 3, 1]),
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            blocks.Permute([0, 3, 1, 2]),
            blocks.Scale(dim, layer_scale),
            blocks.StochasticDepth(survival_prob)
        )

        self.branch2 = nn.Identity()
        self.combine = blocks.Combine('ADD')

    def forward(self, x):
        return self.combine([self.branch1(x), self.branch2(x)])


class DownsamplingBlock(nn.Sequential):
    def __init__(
        self,
        inp: int,
        oup: int
    ):
        super().__init__(
            blocks.LayerNorm2d(inp, eps=1e-6),
            nn.Conv2d(inp, oup, kernel_size=2, stride=2)
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
                blocks.LayerNorm2d(dims[0], eps=1e-6)
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
            blocks.LayerNorm2d(dims[-1], eps=1e-6),
            nn.Flatten(1),
            nn.Linear(dims[-1], num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.1.2-convnext-weights/torch-convnext_t-98aeea18.pth')
def convnext_t(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    model = ConvNeXt(layers=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.1.2-convnext-weights/torch-convnext_s-0ebda7c5.pth')
def convnext_s(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    model = ConvNeXt(layers=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
@config(url='https://github.com/ffiirree/cv-models/releases/download/v0.1.2-convnext-weights/torch-convnext_b-1e0fb038.pth')
def convnext_b(pretrained: bool = False, in_22k=False, pth: str = None, progress: bool = True, **kwargs: Any):
    model = ConvNeXt(layers=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
def convnext_l(pretrained: bool = False, in_22k=False, pth: str = None, progress: bool = True, **kwargs: Any):
    model = ConvNeXt(layers=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
def convnext_xl(pretrained: bool = False, in_22k=False, pth: str = None, progress: bool = True, **kwargs: Any):
    model = ConvNeXt(layers=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model
