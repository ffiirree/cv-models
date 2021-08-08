import os
import torch
import torch.nn as nn
from .core import blocks

__all__ = ['OneNet', 'onenet', 'OneNetv2', 'onenet_v2']


def onenet(pretrained: bool = False, pth: str = None):
    model = OneNet()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class PickLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f'{x.dim()} != 4'
        return torch.cat([
            x[:, :, 0::2, 0::2],
            x[:, :, 1::2, 0::2],
            x[:, :, 0::2, 1::2],
            x[:, :, 1::2, 1::2]], dim=1)


class OneNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            PickLayer(),
            blocks.Conv2d1x1Block(in_channels * 4, filters),
            blocks.Conv2d1x1Block(filters, filters),
            blocks.Conv2d1x1Block(filters, filters),
            PickLayer(),
            blocks.Conv2d1x1Block(filters * 4, filters * 4),
            blocks.Conv2d1x1Block(filters * 4, filters * 4),
            PickLayer(),
            blocks.Conv2d1x1Block(filters * 4 * 4, filters * 4 * 2),
            blocks.Conv2d1x1Block(filters * 4 * 2, filters * 4 * 2),
            blocks.Conv2d1x1Block(filters * 4 * 2, filters * 4),
            PickLayer(),
            blocks.Conv2d1x1Block(filters * 4 * 4, filters * 4 * 4),
            blocks.Conv2d1x1Block(filters * 4 * 4, filters * 4 * 4),
            PickLayer(),
            blocks.Conv2d1x1Block(filters * 4 * 4 * 4, filters * 4 * 4 * 2),
            blocks.Conv2d1x1Block(filters * 4 * 4 * 2, filters * 4 * 4 * 2),
            blocks.Conv2d1x1Block(filters * 4 * 4 * 2, filters * 4 * 4 * 2),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 4 * 4 * 2, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def onenet_v2(pretrained: bool = False, pth: str = None):
    model = OneNetv2()
    if pretrained and pth is not None:
        model.load_state_dict(torch.load(os.path.expanduser(pth)))
    return model


class OneNetv2(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            PickLayer(),
            blocks.Conv2d1x1Block(in_channels * 4, filters),
            blocks.DepthwiseConv2d(filters, filters),
            blocks.Conv2d1x1Block(filters, filters),
            blocks.Conv2d1x1Block(filters, filters),
            PickLayer(),
            blocks.Conv2d1x1Block(filters * 4, filters * 4),
            blocks.DepthwiseConv2d(filters * 4, filters * 4),
            blocks.Conv2d1x1Block(filters * 4, filters * 4),
            PickLayer(),
            blocks.Conv2d1x1Block(filters * 16, filters * 8),
            blocks.DepthwiseConv2d(filters * 8, filters * 8),
            blocks.Conv2d1x1Block(filters * 8, filters * 4),
            blocks.Conv2d1x1Block(filters * 4, filters * 4),
            PickLayer(),
            blocks.Conv2d1x1Block(filters * 16, filters * 16),
            blocks.DepthwiseConv2d(filters * 16, filters * 16),
            blocks.Conv2d1x1Block(filters * 16, filters * 16),
            PickLayer(),
            blocks.Conv2d1x1Block(filters * 64, filters * 32),
            blocks.DepthwiseConv2d(filters * 32, filters * 32),
            blocks.Conv2d1x1Block(filters * 32, filters * 32),
            blocks.Conv2d1x1Block(filters * 32, filters * 32),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 32, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
