import torch.nn as nn
from ..core import SegmentationModel, export, load_from_local_or_url, get_out_channels
import cvm.models as models
from typing import Any


class FCNHead(nn.Sequential):
    def __init__(
        self,
        in_channels: int = 2048,
        channels: int = None,
        num_classes: int = 32,
        dropout_rate: float = 0.1,
    ):
        channels = channels or int(in_channels / 4.0)
        super().__init__(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(channels, num_classes, 1)
        )


@export
class FCN(SegmentationModel):
    ...

@export
def create_fcn(
    backbone: str = 'resnet50_v1',
    num_classes: int = 21,
    aux_loss: bool = False,
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
    decode_head = FCNHead(get_out_channels(backbone.stage4), None, num_classes, dropout_rate)

    model = FCN(backbone, [3, 4] if aux_loss else [4], decode_head, aux_head)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
def fcn_resnet50_v1(*args, **kwargs: Any):
    return create_fcn('resnet50_v1', *args, **kwargs)


@export
def fcn_mobilenet_v3_small(*args, **kwargs: Any):
    return create_fcn('mobilenet_v3_small', *args, **kwargs)


@export
def fcn_mobilenet_v3_large(*args, **kwargs: Any):
    return create_fcn('mobilenet_v3_large', *args, **kwargs)


@export
def fcn_regnet_x_400mf(*args, **kwargs: Any):
    return create_fcn('regnet_x_400mf', *args, **kwargs)
