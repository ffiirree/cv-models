import torch.nn as nn
from ..core import blocks, export, load_from_local_or_url, get_out_channels
import cvm.models as models
from typing import Any, List


@export
class YOLOv1(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        grid_size: List[int] = (7, 7),
        num_boxes_per_cell: int = 2,
        num_classes: int = 20
    ):
        super().__init__()

        self.backbone = backbone

        self.pool = nn.AdaptiveAvgPool2d((7, 7))

        self.head = nn.Sequential(
            blocks.Conv2dBlock(get_out_channels(backbone), 512, 3),
            blocks.Conv2d1x1(512, num_classes + 5 * num_boxes_per_cell)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.head(x)
        return x


def create_yolov1(
    backbone: str = 'resnet50_v1',
    num_classes: int = 21,
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
        **kwargs
    ).features

    model = YOLOv1(backbone, num_classes=num_classes)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


@export
def yolov1_resnet18_v1(
    num_classes: int = 21,
    pretrained_backbone: bool = False,
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    return create_yolov1('resnet18_v1', num_classes, pretrained_backbone, pretrained, pth, progress, **kwargs)


@export
def yolov1_mobilenet_v3_large(
    num_classes: int = 21,
    pretrained_backbone: bool = False,
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    return create_yolov1('mobilenet_v3_large', num_classes, pretrained_backbone, pretrained, pth, progress, **kwargs)


@export
def yolov1_regnet_x_400mf(
    num_classes: int = 21,
    pretrained_backbone: bool = False,
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    return create_yolov1('regnet_x_400mf', num_classes, pretrained_backbone, pretrained, pth, progress, **kwargs)
