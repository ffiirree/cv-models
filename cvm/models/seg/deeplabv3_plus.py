from typing import Any
import torch
import torch.nn as nn
from cvm import models
from ..core import blocks, export, SegmentationModel, get_out_channels, load_from_local_or_url
from torch.nn import functional as F
from .heads import FCNHead, ClsHead


class DeepLabPlusHead(nn.Module):
    def __init__(
        self,
        aspp_in_channels: int,
        feautes_channels: int,
        out_channels: int = 256,
        num_classes: int = 32,
    ):
        super().__init__()

        self.aspp = blocks.ASPP(aspp_in_channels, out_channels, [12, 24, 36])
        self.cat = blocks.Combine('CONCAT')

        self.conv3x3 = blocks.Conv2d3x3(out_channels + feautes_channels, num_classes)

    def forward(self, x, low_level_feautes):
        size = low_level_feautes.shape[-2:]
        aspp_features = self.aspp(x)
        aspp_features = F.interpolate(aspp_features, size=size, mode="bilinear", align_corners=False)
        features = self.cat([aspp_features, low_level_feautes])
        features = self.conv3x3(features)

        return features


@export
class DeepLabV3Plus(SegmentationModel):
    def forward(self, x):
        size = x.shape[-2:]

        stages = self.backbone(x)

        out = self.decode_head(stages[f'stage{self.out_stages[-1]}'], stages[f'stage{self.out_stages[0]}'], )
        out = self.interpolate(out, size=size)

        res = {'out': out}

        if self.aux_head:
            aux = self.aux_head(stages[f'stage{self.out_stages[-2]}'])
            aux = self.interpolate(aux, size=size)
            res['aux'] = aux

        if self.cls_head:
            cls = self.cls_head(stages[f'stage{self.out_stages[-1]}'])
            cls = cls.reshape(cls.shape[0], cls.shape[1], 1, 1)
            res['out'] = out * torch.sigmoid(cls)

        return res


@export
def create_deeplabv3_plus(
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
    decode_head = DeepLabPlusHead(get_out_channels(backbone.stage4),
                                  get_out_channels(backbone.stage2), num_classes=num_classes)

    model = DeepLabV3Plus(backbone, [2, 3, 4] if aux_loss else [2, 4], decode_head, aux_head, cls_head)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
def deeplabv3_plus_resnet50_v1(*args, **kwargs: Any):
    return create_deeplabv3_plus('resnet50_v1', *args, **kwargs)


@export
def deeplabv3_plus_mobilenet_v3_small(*args, **kwargs: Any):
    return create_deeplabv3_plus('mobilenet_v3_small', *args, **kwargs)


@export
def deeplabv3_plus_mobilenet_v3_large(*args, **kwargs: Any):
    return create_deeplabv3_plus('mobilenet_v3_large', *args, **kwargs)


@export
def deeplabv3_plus_regnet_x_400mf(*args, **kwargs: Any):
    return create_deeplabv3_plus('regnet_x_400mf', *args, **kwargs)


@export
def deeplabv3_plus_mobilenet_v1_x1_0(*args, **kwargs: Any):
    return create_deeplabv3_plus('mobilenet_v1_x1_0', *args, **kwargs)


@export
def deeplabv3_plus_sd_mobilenet_v1_x1_0(*args, **kwargs: Any):
    return create_deeplabv3_plus('sd_mobilenet_v1_x1_0', *args, **kwargs)


@export
def deeplabv3_plus_mobilenet_v2_x1_0(*args, **kwargs: Any):
    return create_deeplabv3_plus('mobilenet_v2_x1_0', *args, **kwargs)


@export
def deeplabv3_plus_sd_mobilenet_v2_x1_0(*args, **kwargs: Any):
    return create_deeplabv3_plus('sd_mobilenet_v2_x1_0', *args, **kwargs)


@export
def deeplabv3_plus_shufflenet_v2_x2_0(*args, **kwargs: Any):
    return create_deeplabv3_plus('shufflenet_v2_x2_0', *args, **kwargs)


@export
def deeplabv3_plus_sd_shufflenet_v2_x2_0(*args, **kwargs: Any):
    return create_deeplabv3_plus('sd_shufflenet_v2_x2_0', *args, **kwargs)


@export
def deeplabv3_plus_efficientnet_b0(*args, **kwargs: Any):
    return create_deeplabv3_plus('efficientnet_b0', *args, **kwargs)


@export
def deeplabv3_plus_sd_efficientnet_b0(*args, **kwargs: Any):
    return create_deeplabv3_plus('sd_efficientnet_b0', *args, **kwargs)
