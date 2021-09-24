"""Contains definitions for post- and pre-activation forms of ResNet and ResNet-RS models.
Residual networks (ResNets) were proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
[3] Irwan Bello, William Fedus, Xianzhi Du, Ekin D. Cubuk, Aravind Srinivas,
Tsung-Yi Lin, Jonathon Shlens, Barret Zoph
    Revisiting ResNets: Improved Training and Scaling Strategies.
    arXiv:2103.07579
"""

import os
import torch
import torch.nn as nn
from torch.nn.functional import dropout
from .core import blocks
from typing import Any, List

__all__ = ['ResNet',
           'resnet18_v1', 'resnet34_v1', 'resnet50_v1', 'resnet101_v1', 'resnet152_v1',
           'resnet18_v1d', 'resnet34_v1d', 'resnet50_v1d', 'resnet101_v1d', 'resnet152_v1d',
           'resnet18_v2', 'resnet34_v2', 'resnet50_v2', 'resnet101_v2', 'resnet152_v2',
           'se_resnet18_v1', 'se_resnet34_v1',
           'se_resnet50_v1', 'se_resnet101_v1', 'se_resnet152_v1',
           'se_resnet18_v2', 'se_resnet34_v2',
           'se_resnet50_v2', 'se_resnet101_v2', 'se_resnet152_v2',
           'resnext50_32x4d', 'resnext101_32x8d',
           'resnet_rs_50', 'resnet_rs_101', 'resnet_rs_152', 'resnet_rs_200',
           'resnet_rs_270', 'resnet_rs_350_i256', 'resnet_rs_350_i320', 'resnet_rs_420'
           ]


model_urls = {
    'resnet18_v1': None,
    'resnet34_v1': None,
    'resnet50_v1': None,
    'resnet101_v1': None,
    'resnet152_v1': None,
    'resnet18_v1d': None,
    'resnet34_v1d': None,
    'resnet50_v1d': None,
    'resnet101_v1d': None,
    'resnet152_v1d': None,
    'resnet18_v2': None,
    'resnet34_v2': None,
    'resnet50_v2': None,
    'resnet101_v2': None,
    'resnet152_v2': None,
    'se_resnet18_v1': None,
    'se_resnet34_v1': None,
    'se_resnet50_v1': None,
    'se_resnet101_v1': None,
    'se_resnet152_v1': None,
    'se_resnet18_v2': None,
    'se_resnet34_v2': None,
    'se_resnet50_v2': None,
    'se_resnet101_v2': None,
    'se_resnet152_v2': None,
    'resnext50_32x4d': None,
    'resnet_rs_50': None,
    'resnet_rs_101': None,
    'resnet_rs_152': None,
    'resnet_rs_200': None,
    'resnet_rs_270': None,
    'resnet_rs_350_i256': None,
    'resnet_rs_350_i320': None,
    'resnet_rs_420': None,
    'resnext101_32x8d': None
}


def _resnet(
    arch: str,
    layers: List[int],
    block: nn.Module,
    se_ratio: float = None,
    pretrained: bool = False,
    pth: str = None,
    progress: bool = False,
    **kwargs: Any
):
    model = ResNet(layers=layers, block=block, se_ratio=se_ratio, **kwargs)

    if pretrained:
        if pth is not None:
            state_dict = torch.load(os.path.expanduser(pth))
        else:
            state_dict = torch.hub.load_state_dict_from_url(
                model_urls[arch],
                progress=progress
            )
        model.load_state_dict(state_dict)
    return model


def resnet18_v1(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _resnet('resnet18_v1', [2, 2, 2, 2], blocks.ResBasicBlockV1, None, pretrained, pth, progress, **kwargs)


def resnet34_v1(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _resnet('resnet34_v1', [3, 4, 6, 3], blocks.ResBasicBlockV1, None, pretrained, pth, progress, **kwargs)


def resnet50_v1(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _resnet('resnet50_v1', [3, 4, 6, 3], blocks.BottleneckV1, None, pretrained, pth, progress, **kwargs)


def resnet101_v1(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _resnet('resnet101_v1', [3, 4, 23, 3], blocks.BottleneckV1, None, pretrained, pth, progress, **kwargs)


def resnet152_v1(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _resnet('resnet152_v1', [3, 8, 36, 3], blocks.BottleneckV1, None, pretrained, pth, progress, **kwargs)


def resnet18_v1d(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['use_resnetc_stem'] = True
    kwargs['use_resnetd_shortcut'] = True
    return _resnet('resnet18_v1', [2, 2, 2, 2], blocks.ResBasicBlockV1, None, pretrained, pth, progress, **kwargs)


def resnet34_v1d(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['use_resnetc_stem'] = True
    kwargs['use_resnetd_shortcut'] = True
    return _resnet('resnet34_v1', [3, 4, 6, 3], blocks.ResBasicBlockV1, None, pretrained, pth, progress, **kwargs)


def resnet50_v1d(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['use_resnetc_stem'] = True
    kwargs['use_resnetd_shortcut'] = True
    return _resnet('resnet50_v1', [3, 4, 6, 3], blocks.BottleneckV1, None, pretrained, pth, progress, **kwargs)


def resnet101_v1d(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['use_resnetc_stem'] = True
    kwargs['use_resnetd_shortcut'] = True
    return _resnet('resnet101_v1', [3, 4, 23, 3], blocks.BottleneckV1, None, pretrained, pth, progress, **kwargs)


def resnet152_v1d(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['use_resnetc_stem'] = True
    kwargs['use_resnetd_shortcut'] = True
    return _resnet('resnet152_v1', [3, 8, 36, 3], blocks.BottleneckV1, None, pretrained, pth, progress, **kwargs)


def resnet18_v2(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _resnet('resnet18_v2', [2, 2, 2, 2], blocks.ResBasicBlockV2, None, pretrained, pth, progress, **kwargs)


def resnet34_v2(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _resnet('resnet34_v2', [3, 4, 6, 3], blocks.ResBasicBlockV2, None, pretrained, pth, progress, **kwargs)


def resnet50_v2(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _resnet('resnet50_v2', [3, 4, 6, 3], blocks.BottleneckV2, None, pretrained, pth, progress, **kwargs)


def resnet101_v2(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _resnet('resnet101_v2', [3, 4, 23, 3], blocks.BottleneckV2, None, pretrained, pth, progress, **kwargs)


def resnet152_v2(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _resnet('resnet152_v2', [3, 8, 36, 3], blocks.BottleneckV2, None, pretrained, pth, progress, **kwargs)


def se_resnet18_v1(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _resnet('se_resnet18_v1', [2, 2, 2, 2], blocks.ResBasicBlockV1, None, pretrained, pth, progress, **kwargs)


def se_resnet34_v1(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _resnet('se_resnet34_v1', [3, 4, 6, 3], blocks.ResBasicBlockV1, None, pretrained, pth, progress, **kwargs)


def se_resnet50_v1(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _resnet('se_resnet50_v1', [3, 4, 6, 3], blocks.BottleneckV1, 1/16, pretrained, pth, progress, **kwargs)


def se_resnet101_v1(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _resnet('se_resnet101_v1', [3, 4, 23, 3], blocks.BottleneckV1, 1/16, pretrained, pth, progress, **kwargs)


def se_resnet152_v1(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _resnet('resnet152_v1', [3, 8, 36, 3], blocks.BottleneckV1, 1/16, pretrained, pth, progress, **kwargs)


def se_resnet18_v2(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _resnet('resnet18_v1', [2, 2, 2, 2], blocks.ResBasicBlockV2, 1/16, pretrained, pth, progress, **kwargs)


def se_resnet34_v2(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _resnet('se_resnet34_v2', [3, 4, 6, 3], blocks.ResBasicBlockV2, 1/16, pretrained, pth, progress, **kwargs)


def se_resnet50_v2(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _resnet('se_resnet50_v2', [3, 4, 6, 3], blocks.BottleneckV2, 1/16, pretrained, pth, progress, **kwargs)


def se_resnet101_v2(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _resnet('se_resnet101_v2', [3, 4, 23, 3], blocks.BottleneckV2, 1/16, pretrained, pth, progress, **kwargs)


def se_resnet152_v2(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    return _resnet('se_resnet152_v2', [3, 8, 36, 3], blocks.BottleneckV2, 1/16, pretrained, pth, progress, **kwargs)


def resnext50_32x4d(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnet152', [3, 4, 6, 3], blocks.BottleneckV1, None, pretrained, pth, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnet152', [3, 4, 23, 3], blocks.BottleneckV1, None, pretrained, pth, progress, **kwargs)


def resnet_rs_50(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['replace_stem_max_pool'] = True
    kwargs['use_resnetc_stem'] = True
    kwargs['use_resnetd_shortcut'] = True
    kwargs['dropout_rate'] = 0.25
    return _resnet('resnet_rs_50', [3, 4, 6, 3], blocks.BottleneckV1, 0.25, pretrained, pth, progress, **kwargs)


def resnet_rs_101(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['replace_stem_max_pool'] = True
    kwargs['use_resnetc_stem'] = True
    kwargs['use_resnetd_shortcut'] = True
    kwargs['dropout_rate'] = 0.25
    return _resnet('resnet_rs_101', [3, 4, 23, 3], blocks.BottleneckV1, 0.25, pretrained, pth, progress, **kwargs)


def resnet_rs_152(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['replace_stem_max_pool'] = True
    kwargs['use_resnetc_stem'] = True
    kwargs['use_resnetd_shortcut'] = True
    kwargs['dropout_rate'] = 0.25
    return _resnet('resnet_rs_152', [3, 8, 36, 3], blocks.BottleneckV1, 0.25, pretrained, pth, progress, **kwargs)


def resnet_rs_200(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['replace_stem_max_pool'] = True
    kwargs['use_resnetc_stem'] = True
    kwargs['use_resnetd_shortcut'] = True
    kwargs['drop_connect_rate'] = 0.1
    kwargs['dropout_rate'] = 0.25
    return _resnet('resnet_rs_200', [3, 24, 36, 3], blocks.BottleneckV1, 0.25, pretrained, pth, progress, **kwargs)


def resnet_rs_270(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['replace_stem_max_pool'] = True
    kwargs['use_resnetc_stem'] = True
    kwargs['use_resnetd_shortcut'] = True
    kwargs['drop_connect_rate'] = 0.1
    kwargs['dropout_rate'] = 0.25
    return _resnet('resnet_rs_270', [4, 29, 53, 4], blocks.BottleneckV1, 0.25, pretrained, pth, progress, **kwargs)


def resnet_rs_350_i256(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['replace_stem_max_pool'] = True
    kwargs['use_resnetc_stem'] = True
    kwargs['use_resnetd_shortcut'] = True
    kwargs['drop_connect_rate'] = 0.1
    kwargs['dropout_rate'] = 0.25
    return _resnet('resnet_rs_350', [4, 4, 72, 4], blocks.BottleneckV1, 0.25, pretrained, pth, progress, **kwargs)


def resnet_rs_350_i320(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['replace_stem_max_pool'] = True
    kwargs['use_resnetc_stem'] = True
    kwargs['use_resnetd_shortcut'] = True
    kwargs['drop_connect_rate'] = 0.1
    kwargs['dropout_rate'] = 0.4
    return _resnet('resnet_rs_350', [4, 4, 72, 4], blocks.BottleneckV1, 0.25, pretrained, pth, progress, **kwargs)


def resnet_rs_420(pretrained: bool = False, pth: str = None, progress: bool = True, **kwargs: Any):
    kwargs['replace_stem_max_pool'] = True
    kwargs['use_resnetc_stem'] = True
    kwargs['use_resnetd_shortcut'] = True
    kwargs['drop_connect_rate'] = 0.1
    kwargs['dropout_rate'] = 0.4
    return _resnet('resnet_rs_420', [4, 4, 87, 4], blocks.BottleneckV1, 0.25, pretrained, pth, progress, **kwargs)

# TODO: drop_connect_rate


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        layers: List[int] = [2, 2, 2, 2],
        groups: int = 1,
        width_per_group: int = 64,
        se_ratio: float = None,
        dropout_rate: float = None,
        drop_connect_rate: float = None,
        block: nn.Module = blocks.ResBasicBlockV1,
        thumbnail: bool = False,
        replace_stem_max_pool: bool = False,
        use_resnetc_stem: bool = False,
        use_resnetd_shortcut: bool = False
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 2

        self.layers = layers
        self.groups = groups
        self.width_per_group = width_per_group
        self.block = block
        self.ratio = se_ratio
        self.drop_connect_rate = drop_connect_rate
        self.use_resnetd_shortcut = use_resnetd_shortcut
        self.version = 1
        if issubclass(block, (blocks.ResBasicBlockV2, blocks.BottleneckV2)):
            self.version = 2

        if use_resnetc_stem:
            features = [
                blocks.Conv2d3x3(in_channels, 64, stride=FRONT_S),
                *blocks.norm_activation(64),
                blocks.Conv2d3x3(64, 64),
                *blocks.norm_activation(64),
                blocks.Conv2d3x3(64, 64)
            ]
        else:
            features = [
                nn.Conv2d(in_channels, 64, 7, FRONT_S, padding=3, bias=False)
            ]

        if self.version == 1 or replace_stem_max_pool:
            features.extend(blocks.norm_activation(64))

        if replace_stem_max_pool:
            features.append(blocks.Conv2d3x3(64, 64, stride=FRONT_S))
            if self.version == 1:
                features.extend(blocks.norm_activation(64))
        elif not thumbnail:
            features.append(nn.MaxPool2d(3, stride=2, padding=1))

        features.extend(self.make_layers(
            64 // block.expansion, 64, 1, layers[0], 2))
        features.extend(self.make_layers(64, 128, 2, layers[1], 3))
        features.extend(self.make_layers(128, 256, 2, layers[2], 4))
        features.extend(self.make_layers(256, 512, 2, layers[3], 5))

        if self.version == 2:
            features.extend(blocks.norm_activation(512 * self.block.expansion))

        self.features = nn.Sequential(*features)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential()
        if dropout_rate is not None:
            self.classifier.add_module('do', nn.Dropout(dropout_rate))
        self.classifier.add_module('fc', nn.Linear(
            512 * block.expansion, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_drop_connect_rate(self, block_num: int):
        if self.drop_connect_rate is not None:
            return self.drop_connect_rate * float(block_num) / (len(self.layers) + 1)
        else:
            return None

    def make_layers(self, inp, oup, stride, n, block_num):
        layers = [
            self.block(
                inp * self.block.expansion,
                oup,
                stride=stride,
                groups=self.groups,
                width_per_group=self.width_per_group,
                se_ratio=self.ratio,
                drop_connect_rate=self.get_drop_connect_rate(block_num),
                use_resnetd_shortcut=self.use_resnetd_shortcut
            )
        ]
        for _ in range(n - 1):
            layers.append(
                self.block(
                    oup * self.block.expansion,
                    oup,
                    groups=self.groups,
                    width_per_group=self.width_per_group,
                    se_ratio=self.ratio,
                    drop_connect_rate=self.get_drop_connect_rate(block_num),
                    use_resnetd_shortcut=self.use_resnetd_shortcut
                )
            )
        return layers
