import math
import torch
import torch.nn as nn
from .core import blocks

__all__ = ['EfficientNetv2', 'efficientnet_v2_s',
           'efficientnet_v2_m', 'efficientnet_v2_l', 'efficientnet_v2_xl']
_BN_EPSILON = 1e-3
# Paper suggests 0.99 momentum
_BN_MOMENTUM = 0.01

blocks.BN_EPSILON = _BN_EPSILON
blocks.BN_MOMENTUM = _BN_MOMENTUM


def efficientnet_params(model_name):
    """Get efficientnet params based on model name."""
    params_dict = {
        # (width_coefficient, depth_coefficient, resolution, dropout_rate)
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]


efficientnetv2_params = {
    # (width, depth, train_size, eval_size, dropout, randaug, mixup, aug)
    # 83.9% @ 22M
    'efficientnetv2-s': (1.0, 1.0, 300, 384, 0.2, 10, 0, 'randaug'),
    # 85.2% @ 54M
    'efficientnetv2-m': (1.0, 1.0, 384, 480, 0.3, 15, 0.2, 'randaug'),
    # 85.7% @ 120M
    'efficientnetv2-l': (1.0, 1.0, 384, 480, 0.4, 20, 0.5, 'randaug'),
    'efficientnetv2-xl': (1.0, 1.0, 384, 512, 0.4, 20, 0.5, 'randaug'),
}


def efficientnet_v2_s(pretrained: bool = False):
    return EfficientNetv2(
        dropout_rate=0.2,
        block_type=[0, 0, 0, 1, 1, 1],
        expand_ratio=[1, 4, 4, 4, 6, 6],
        filters=[24, 24, 48, 64, 128, 160, 256, 1280],
        layers=[2, 4, 5, 6, 9, 15],
        strides=[1, 2, 2, 2, 1, 2],
        se_ratio=[0, 0, 0, 0.25, 0.25, 0.25]
    )


def efficientnet_v2_m(pretrained: bool = False):
    return EfficientNetv2(
        dropout_rate=0.3,
        block_type=[0, 0, 0, 1, 1, 1, 1],
        expand_ratio=[1, 4, 4, 4, 6, 6, 6],
        filters=[24, 24, 48, 80, 160, 176, 304, 512, 1280],
        layers=[3, 5, 5, 7, 14, 18, 5],
        strides=[1, 2, 2, 2, 1, 2, 1],
        se_ratio=[0, 0, 0, 0.25, 0.25, 0.25, 0.25]
    )


def efficientnet_v2_l(pretrained: bool = False):
    return EfficientNetv2(
        dropout_rate=0.3,
        block_type=[0, 0, 0, 1, 1, 1, 1],
        expand_ratio=[1, 4, 4, 4, 6, 6, 6],
        filters=[32, 32, 64, 96, 192, 224, 384, 640, 1280],
        layers=[4, 7, 7, 10, 19, 25, 7],
        strides=[1, 2, 2, 2, 1, 2, 1],
        se_ratio=[0, 0, 0, 0.25, 0.25, 0.25, 0.25]
    )


def efficientnet_v2_xl(pretrained: bool = False):
    return EfficientNetv2(
        dropout_rate=0.4,
        block_type=[0, 0, 0, 1, 1, 1, 1],
        expand_ratio=[1, 4, 4, 4, 6, 6, 6],
        filters=[32, 32, 64, 96, 192, 256, 512, 640, 1280],
        layers=[4, 8, 8, 16, 24, 32, 8],
        strides=[1, 2, 2, 2, 1, 2, 1],
        se_ratio=[0, 0, 0, 0.25, 0.25, 0.25, 0.25]
    )


class EfficientNetv2(nn.Module):

    activation_layer = nn.SiLU

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        dropout_rate: float = 0.2,
        block_type: list = [0, 0, 0, 1, 1, 1],
        expand_ratio: list = [1, 4, 4, 4, 6, 6],
        filters: list = [24, 24, 48, 64, 128, 160, 256, 1280],
        layers: list = [2, 4, 5, 6, 9, 15],
        strides: list = [1, 2, 2, 2, 1, 2],
        se_ratio: list = [0, 0, 0, 0.25, 0.25, 0.25],
    ):
        super().__init__()

        self.survival_prob = 0.8
        self.dropout_rate = dropout_rate
        self.blocks = sum(layers)
        self.block_idx = 0

        features = [
            blocks.Conv2dBlock(
                in_channels, filters[0], stride=2, activation_layer=self.activation_layer)
        ]

        for i in range(len(expand_ratio)):
            features.append(
                self.make_layers(
                    block_type[i], filters[i], expand_ratio[i], filters[i+1],
                    n=layers[i], stride=strides[i], se_ratio=se_ratio[i]
                )
            )

        features.append(blocks.Conv2d1x1Block(filters[-2], filters[-1]))

        self.features = nn.Sequential(*features)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(filters[-1], num_classes)
        )

    def make_layers(
        self,
        block_type: int,
        inp: int,
        t: int,
        oup: int,
        n: int,
        stride: int,
        se_ratio: float = None
    ):
        layers = []
        for i in range(n):
            block = blocks.InvertedResidualBlock if block_type == 1 else blocks.FusedInvertedResidualBlock
            inp = inp if i == 0 else oup
            stride = stride if i == 0 else 1
            survival_prob = self.survival_prob + \
                (1 - self.survival_prob) * (i + self.block_idx) / self.blocks

            layers.append(
                block(
                    inp, oup, t,
                    stride=stride,
                    survival_prob=survival_prob, se_ratio=se_ratio,
                    activation_layer=self.activation_layer
                )
            )

        self.block_idx += n

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
