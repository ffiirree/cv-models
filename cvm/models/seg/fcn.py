from ..core import SegmentationModel, export, load_from_local_or_url, get_out_channels
from .heads import ClsHead, FCNHead
import cvm.models as models
from typing import Any


@export
class FCN(SegmentationModel):
    ...


@export
def create_fcn(
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
    decode_head = FCNHead(get_out_channels(backbone.stage4), None, num_classes, dropout_rate)

    model = FCN(backbone, [3, 4] if aux_loss else [4], decode_head, aux_head, cls_head)

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
