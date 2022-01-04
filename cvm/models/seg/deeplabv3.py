from typing import Any, List
import torch.nn as nn
from cvm import models
from ..core import blocks, export, SegmentationModel, get_out_channels, load_from_local_or_url
from torch.nn import functional as F
from .heads import FCNHead, ClsHead


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            blocks.Conv2d1x1Block(in_channels, out_channels)
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        rates: List[int] = [6, 12, 18]
    ):
        super().__init__()

        ms = [blocks.Conv2d1x1Block(in_channels, out_channels)]
        for rate in rates:
            ms.append(blocks.Conv2dBlock(in_channels, out_channels, padding=rate, dilation=rate))

        ms.append(ASPPPooling(in_channels, out_channels))
        self.ms = nn.ModuleList(ms)

        self.combine = blocks.Combine('CONCAT')
        self.conv1x1 = blocks.Conv2d1x1(out_channels * len(self.ms), out_channels)

    def forward(self, x):
        aspp = []
        for module in self.ms:
            aspp.append(module(x))

        x = self.combine(aspp)
        x = self.conv1x1(x)
        return x


class DeepLabHead(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        num_classes: int = 32,
    ):
        super().__init__(
            ASPP(in_channels, out_channels, [12, 24, 36]),
            blocks.Conv2dBlock(out_channels, out_channels),
            blocks.Conv2d1x1(out_channels, num_classes)
        )


@export
class DeepLabV3(SegmentationModel):
    ...


@export
def create_deeplabv3(
    backbone: str = 'resnet50_v1',
    num_classes: int = 21,
    aux_loss: bool = False,
    cls_loss: bool = False,
    dropout_rate: float = 0.1,
    pretrained_backbone: bool = False,
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    if pretrained:
        pretrained_backbone = False

    backbone = models.__dict__[backbone](
        pretrained=pretrained_backbone,
        dilations=[1, 1, 2, 4],
        **kwargs
    ).features

    aux_head = FCNHead(get_out_channels(backbone.stage3), None, num_classes, dropout_rate) if aux_loss else None
    cls_head = ClsHead(get_out_channels(backbone.stage4), num_classes) if cls_loss else None
    decode_head = DeepLabHead(get_out_channels(backbone.stage4), num_classes=num_classes)

    model = DeepLabV3(backbone, [3, 4] if aux_loss else [4], decode_head, aux_head, cls_head)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
def deeplabv3_resnet50_v1(*args, **kwargs: Any):
    return create_deeplabv3('resnet50_v1', *args, **kwargs)


@export
def deeplabv3_mobilenet_v3_small(*args, **kwargs: Any):
    return create_deeplabv3('mobilenet_v3_small', *args, **kwargs)


@export
def deeplabv3_mobilenet_v3_large(*args, **kwargs: Any):
    return create_deeplabv3('mobilenet_v3_large', *args, **kwargs)


@export
def deeplabv3_regnet_x_400mf(*args, **kwargs: Any):
    return create_deeplabv3('regnet_x_400mf', *args, **kwargs)
