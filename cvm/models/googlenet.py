import torch
import torch.nn as nn

from .ops import blocks
from .utils import export, load_from_local_or_url
from typing import Any, List, OrderedDict

__all__ = ['inception_v1']


class InceptionBlock(blocks.ConcatBranches):
    def __init__(
        self,
        inp,
        planes_1x1: int,
        planes_3x3: List[int],
        planes_5x5: List[int],
        planes_pool: int
    ):
        super().__init__(OrderedDict([
            ('branch-1x1', blocks.Conv2d1x1Block(inp, planes_1x1)),
            ('branch-3x3', nn.Sequential(
                blocks.Conv2d1x1Block(inp, planes_3x3[0]),
                blocks.Conv2dBlock(planes_3x3[0], planes_3x3[1])
            )),
            ('branch-5x5', nn.Sequential(
                blocks.Conv2d1x1Block(inp, planes_5x5[0]),
                blocks.Conv2dBlock(planes_5x5[0], planes_5x5[1], kernel_size=5, padding=2)
            )),
            ('branch-pool', nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                blocks.Conv2d1x1Block(inp, planes_pool)
            ))
        ]))


class InceptionAux(nn.Sequential):
    def __init__(self, inp, oup):
        super().__init__(
            nn.AdaptiveAvgPool2d((4, 4)),
            blocks.Conv2d1x1Block(inp, 128),
            nn.Flatten(1),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(1024, oup)
        )


@export
def googlenet(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    model = GoogLeNet(**kwargs)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


inception_v1 = googlenet


@export
class GoogLeNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        thumbnail: bool = False,
        **kwargs: Any
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 2

        self.stem = nn.Sequential(
            blocks.Conv2dBlock(in_channels, 64, 7, stride=FRONT_S, padding=3),
            nn.Identity() if thumbnail else nn.MaxPool2d(3, 2, ceil_mode=True)
        )

        self.stage1 = nn.Sequential(
            blocks.Conv2d1x1Block(64, 64),
            blocks.Conv2dBlock(64, 192, 3, padding=1),
            nn.MaxPool2d(3, 2, ceil_mode=True)
        )

        self.stage2 = nn.Sequential(OrderedDict([
            ('inception_3a', InceptionBlock(192, 64, [96, 128], [16, 32], 32)),
            ('inception_3b', InceptionBlock(256, 128, [128, 192], [32, 96], 64)),
            ('max_pool', nn.MaxPool2d(3, 2, ceil_mode=True))
        ]))

        self.stage3 = nn.Sequential(OrderedDict([
            ('inception_4a', InceptionBlock(480, 192, [96, 208], [16, 48], 64)),
            ('inception_4b', InceptionBlock(512, 160, [112, 224], [24, 64], 64)),
            ('inception_4c', InceptionBlock(512, 128, [128, 256], [24, 64], 64)),
            ('inception_4d', InceptionBlock(512, 112, [144, 288], [32, 64], 64)),
            ('inception_4e', InceptionBlock(528, 256, [160, 320], [32, 128], 128)),
            ('max_pool', nn.MaxPool2d(3, 2, ceil_mode=True))
        ]))

        self.stage4 = nn.Sequential(OrderedDict([
            ('inception_5a', InceptionBlock(832, 256, [160, 320], [32, 128], 128)),
            ('inception_5b', InceptionBlock(832, 384, [192, 384], [48, 128], 128))
        ]))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifiar = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )

        self.aux1 = InceptionAux(512, num_classes)
        self.aux2 = InceptionAux(528, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3.inception_4a(x)
        aux1 = self.aux1(x) if self.training else None
        x = self.stage3.inception_4b(x)
        x = self.stage3.inception_4c(x)
        x = self.stage3.inception_4d(x)
        aux2 = self.aux2(x) if self.training else None
        x = self.stage3.inception_4e(x)

        x = self.stage3.max_pool(x)

        x = self.stage4(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifiar(x)

        if self.training:
            return x, aux1, aux2
        else:
            return x
