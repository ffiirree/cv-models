import torch.nn as nn
from ..core import SegmentationModel, export, load_from_local_or_url
import cvm.models as models
from typing import Any


class FCNHead(nn.Sequential):
    def __init__(
        self,
        in_channels: int = 2048,
        channels: int = 512,
        num_classes: int = 32,
        dropout_rate: float = 0.1,
    ):
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


def _fcn(
    backbone: str = 'resnet50_v1',
    decode_head: nn.Module = None,
    aux_head: nn.Module = None,
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

    model = FCN(backbone, [3, 4], decode_head, aux_head)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
def fcn_resnet50_v1(
    num_classes: int = 21,
    aux_loss: bool = False,
    dropout_rate: float = 0.1,
    pretrained: bool = False,
    pretrained_backbone: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    decode_head = FCNHead(2048, 512, num_classes, dropout_rate)
    aux_head = FCNHead(1024, 256, num_classes, dropout_rate) if aux_loss else None

    return _fcn('resnet50_v1', decode_head, aux_head, pretrained_backbone, pretrained, pth, progress, **kwargs)


@export
def fcn_mobilenet_v3_small(
    num_classes: int = 21,
    aux_loss: bool = False,
    dropout_rate: float = 0.1,
    pretrained: bool = False,
    pretrained_backbone: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    decode_head = FCNHead(576, 512, num_classes, dropout_rate)
    aux_head = FCNHead(48, 12, num_classes, dropout_rate) if aux_loss else None

    return _fcn('mobilenet_v3_small', decode_head, aux_head, pretrained_backbone, pretrained, pth, progress, **kwargs)


@export
def fcn_mobilenet_v3_large(
    num_classes: int = 21,
    aux_loss: bool = False,
    dropout_rate: float = 0.1,
    pretrained: bool = False,
    pretrained_backbone: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    decode_head = FCNHead(960, 512, num_classes, dropout_rate)
    aux_head = FCNHead(112, 28, num_classes, dropout_rate) if aux_loss else None

    return _fcn('mobilenet_v3_large', decode_head, aux_head, pretrained_backbone, pretrained, pth, progress, **kwargs)


@export
def fcn_regnet_x_400mf(
    num_classes: int = 21,
    aux_loss: bool = False,
    dropout_rate: float = 0.1,
    pretrained: bool = False,
    pretrained_backbone: bool = True,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    decode_head = FCNHead(400, 100, num_classes, dropout_rate)
    aux_head = FCNHead(160, 40, num_classes, dropout_rate) if aux_loss else None

    return _fcn('regnet_x_400mf', decode_head, aux_head, pretrained_backbone, pretrained, pth, progress, **kwargs)
