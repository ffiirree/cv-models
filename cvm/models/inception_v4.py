import torch
import torch.nn as nn
from .ops import blocks
from typing import List, Any
from .utils import export, load_from_local_or_url


def get_stem(in_channels):
    return blocks.Stage(
        blocks.Conv2dBlock(in_channels, 32, kernel_size=3, stride=2, padding=0),
        blocks.Conv2dBlock(32, 32, kernel_size=3, padding=0),
        blocks.Conv2dBlock(32, 64, kernel_size=3),
        blocks.ConcatBranches(
            nn.MaxPool2d(3, stride=2),
            blocks.Conv2dBlock(64, 96, kernel_size=3, stride=2, padding=0)
        ),
        blocks.ConcatBranches(
            nn.Sequential(
                blocks.Conv2d1x1Block(160, 64),
                blocks.Conv2dBlock(64, 96, kernel_size=3, padding=0)
            ),
            nn.Sequential(
                blocks.Conv2d1x1Block(160, 64),
                blocks.Conv2dBlock(64, 64, kernel_size=(7, 1), padding=(3, 0)),
                blocks.Conv2dBlock(64, 64, kernel_size=(1, 7), padding=(0, 3)),
                blocks.Conv2dBlock(64, 96, kernel_size=3, padding=0)
            )
        ),
        blocks.ConcatBranches(
            blocks.Conv2dBlock(192, 192, kernel_size=3, stride=2, padding=0),
            nn.MaxPool2d(3, stride=2, padding=0)
        )
    )


class InceptionV4(nn.Module):
    r"""
    Paper: Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning, https://arxiv.org/abs/1602.07261
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        **kwargs: Any
    ) -> None:
        super().__init__()

        self.stem = get_stem(in_channels)

        self.stage1 = blocks.Stage(
            *[blocks.InceptionA(384, 96, [64, 96], [64, 96], 96) for _ in range(4)],
            blocks.ReductionA(384, 384, [192, 224, 256]),
        )

        self.stage2 = blocks.Stage(
            *[blocks.InceptionB(1024, 384, [192, 224, 256], [192, 224, 256], 128) for _ in range(7)],
            blocks.ReductionB(1024, [192, 192], [256, 320])
        )

        self.stage3 = blocks.Stage(
            *[blocks.InceptionC(1536, 256, [384, 256], [384, 448, 512, 256], 256) for _ in range(3)],
        )

        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate, inplace=True),
            nn.Linear(1536, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


@export
def inception_v4(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    model = InceptionV4(**kwargs)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


class InceptionResNetV1(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        **kwargs: Any
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            blocks.Conv2dBlock(in_channels, 32, kernel_size=3, stride=2, padding=0),
            blocks.Conv2dBlock(32, 32, kernel_size=3, padding=0),
            blocks.Conv2dBlock(32, 64, kernel_size=3),
            nn.MaxPool2d(3, stride=2),
            blocks.Conv2d1x1Block(64, 80),
            blocks.Conv2dBlock(80, 192, kernel_size=3, padding=0),
            blocks.Conv2dBlock(192, 256, kernel_size=3, stride=2, padding=0)
        )

        self.stage1 = blocks.Stage(
            *[blocks.InceptionResNetA(256, 32, [32, 32], [32, 32, 32]) for _ in range(5)],
            blocks.ReductionA(256, 384, [192, 192, 256])
        )

        self.stage2 = blocks.Stage(
            *[blocks.InceptionResNetB(896, 128, [128, 128, 128]) for _ in range(10)],
            blocks.ReductionC(896, [256, 384], [256, 256], [256, 256, 256])
        )

        self.stage3 = blocks.Stage(
            [blocks.InceptionResNetC(1792, 192, [192, 192, 192]) for _ in range(5)],
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate, inplace=True),
            nn.Linear(1792, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


@export
def inception_resnet_v1(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    model = InceptionResNetV1(**kwargs)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model


class InceptionResNetV2(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        **kwargs: Any
    ) -> None:
        super().__init__()

        self.stem = get_stem(in_channels)

        self.stage1 = blocks.Stage(
            *[blocks.InceptionResNetA(384, 32, [32, 32], [32, 48, 64]) for _ in range(10)],
            blocks.ReductionA(384, 384, [256, 256, 384])
        )

        self.stage2 = blocks.Stage(
            *[blocks.InceptionResNetB(1152, 192, [128, 160, 192]) for _ in range(20)],
            blocks.ReductionC(1152, [256, 384], [256, 288], [256, 288, 320])
        )

        self.stage3 = blocks.Stage(
            *[blocks.InceptionResNetC(2144, 192, [192, 224, 256]) for _ in range(10)],
            blocks.Conv2d1x1Block(2144, 1536)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate, inplace=True),
            nn.Linear(1536, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


@export
def inception_resnet_v2(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    model = InceptionResNetV2(**kwargs)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model
