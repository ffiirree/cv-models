import torch
import torch.nn as nn
from .ops import blocks
from .utils import export, load_from_local_or_url
from typing import Any, List, OrderedDict


# Figure 5
class InceptionBlockV5(blocks.ConcatBranches):
    def __init__(
        self,
        inp,
        planes_1x1: int,
        planes_5x5: List[int],
        planes_3x3db: List[int],
        planes_pool: int
    ):
        super().__init__(OrderedDict([
            ('branch-1x1', blocks.Conv2d1x1Block(inp, planes_1x1)),
            ('branch-5x5', nn.Sequential(
                blocks.Conv2d1x1Block(inp, planes_5x5[0]),
                blocks.Conv2dBlock(planes_5x5[0], planes_5x5[1], kernel_size=5, padding=2)
            )),
            ('branch-3x3db', nn.Sequential(
                blocks.Conv2d1x1Block(inp, planes_3x3db[0]),
                blocks.Conv2dBlock(planes_3x3db[0], planes_3x3db[1]),
                blocks.Conv2dBlock(planes_3x3db[1], planes_3x3db[1])
            )),
            ('branch-pool', nn.Sequential(
                nn.AvgPool2d(3, stride=1, padding=1),
                blocks.Conv2d1x1Block(inp, planes_pool)
            ))
        ]))


# Figure 6: blocks.InceptionB


# Figure 7
class InceptionBlockV7(blocks.ConcatBranches):
    def __init__(
        self,
        inp,
        planes_1x1: int,
        planes_3x3: List[int],
        planes_3x3db: List[int],
        planes_pool
    ) -> None:
        super().__init__(OrderedDict([
            ('branch_1x1', blocks.Conv2d1x1Block(inp, planes_1x1)),
            ('branch-3x3', nn.Sequential(
                blocks.Conv2d1x1Block(inp, planes_3x3[0]),
                blocks.ConcatBranches(OrderedDict([
                    ('branch-3x3-1', blocks.Conv2dBlock(
                        planes_3x3[0], planes_3x3[1], kernel_size=(1, 3), padding=(0, 1)
                    )),
                    ('branch-3x3-2', blocks.Conv2dBlock(
                        planes_3x3[0], planes_3x3[1], kernel_size=(3, 1), padding=(1, 0)
                    ))
                ]))
            )),
            ('branch-3x3db', nn.Sequential(
                blocks.Conv2d1x1Block(inp, planes_3x3db[0]),
                blocks.Conv2dBlock(planes_3x3db[0], planes_3x3db[1]),
                blocks.ConcatBranches(OrderedDict([
                    ('branch-3x3db-1', blocks.Conv2dBlock(
                        planes_3x3db[1], planes_3x3db[1], kernel_size=(1, 3), padding=(0, 1)
                    )),
                    ('branch-3x3db-2', blocks.Conv2dBlock(
                        planes_3x3db[1], planes_3x3db[1], kernel_size=(3, 1), padding=(1, 0)
                    ))
                ]))
            )),
            ('branch-pool', nn.Sequential(
                nn.AvgPool2d(3, stride=1, padding=1),
                blocks.Conv2d1x1Block(inp, planes_pool)
            ))
        ]))


class InceptionV3(nn.Module):
    r"""
    Paper: Rethinking the Inception Architecture for Computer Vision, https://arxiv.org/abs/1512.00567
    Code: https://github.com/keras-team/keras/blob/master/keras/applications/inception_v3.py
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        dropout_rate: float = 0.2,
        thumbnail: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__()

        self.stem = blocks.Conv2dBlock(in_channels, 32, kernel_size=3, stride=2, padding=0)

        self.stage1 = blocks.Stage(
            blocks.Conv2dBlock(32, 32, kernel_size=3, padding=0),
            blocks.Conv2dBlock(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.stage2 = blocks.Stage(
            blocks.Conv2d1x1Block(64, 80),
            blocks.Conv2dBlock(80, 192, kernel_size=3, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.stage3 = blocks.Stage(
            InceptionBlockV5(192, 64, [48, 64], [64, 96], 32),                      # mix 0: 35 x 35 x 256
            InceptionBlockV5(256, 64, [48, 64], [64, 96], 64),                      # mix 1: 35 x 35 x 288
            InceptionBlockV5(288, 64, [48, 64], [64, 96], 64),                      # mix 2: 35 x 35 x 288
            blocks.ReductionA(288, 384, [64, 96, 96])                               # mix 3: 17 x 17 x 768
        )

        self.stage4 = blocks.Stage(
            blocks.InceptionB(768, 192, [128, 128, 192], [128, 128, 192], 192),     # mix 4: 17 x 17 x 768
            blocks.InceptionB(768, 192, [160, 160, 192], [160, 160, 192], 192),     # mix 5: 17 x 17 x 768
            blocks.InceptionB(768, 192, [160, 160, 192], [160, 160, 192], 192),     # mix 6: 17 x 17 x 768
            blocks.InceptionB(768, 192, [192, 192, 192], [192, 192, 192], 192),     # mix 7: 17 x 17 x 768
            blocks.ReductionB(768, [192, 320], [192, 192])                          # mix 8: 17 x 17 x 1280
        )

        self.stage5 = blocks.Stage(
            InceptionBlockV7(1280, 320, [384, 384], [448, 384], 192),               # mixed 9: 8 x 8 x 2048
            InceptionBlockV7(2048, 320, [384, 384], [448, 384], 192),               # mixed 9: 8 x 8 x 2048
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifer = nn.Sequential(
            nn.Dropout(dropout_rate, inplace=True),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifer(x)
        return x


@export
def inception_v3(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    model = InceptionV3(**kwargs)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model
