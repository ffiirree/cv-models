import torch
import torch.nn as nn

__all__ = ['OneNet', 'OneNetv2', 'OneNetv3']


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

class Conv2d1x1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.layer(x)

class OneNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, base_filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            PickLayer(),
            Conv2d1x1(in_channels * 4, base_filters),
            Conv2d1x1(base_filters, base_filters),
            Conv2d1x1(base_filters, base_filters),
            PickLayer(),
            Conv2d1x1(base_filters * 4, base_filters * 4),
            Conv2d1x1(base_filters * 4, base_filters * 4),
            PickLayer(),
            Conv2d1x1(base_filters * 4 * 4, base_filters * 4 * 2),
            Conv2d1x1(base_filters * 4 * 2, base_filters * 4 * 2),
            Conv2d1x1(base_filters * 4 * 2, base_filters * 4),
            PickLayer(),
            Conv2d1x1(base_filters * 4 * 4, base_filters * 4 * 4),
            Conv2d1x1(base_filters * 4 * 4, base_filters * 4 * 4),
            PickLayer(),
            Conv2d1x1(base_filters * 4 * 4 * 4, base_filters * 4 * 4 * 2),
            Conv2d1x1(base_filters * 4 * 4 * 2, base_filters * 4 * 4 * 2),
            Conv2d1x1(base_filters * 4 * 4 * 2, base_filters * 4 * 4 * 2),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_filters * 4 * 4 * 2, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class DWBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size:int = 3, stride: int = 1, padding: int = 1):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, bias=False, stride=stride, padding=padding, groups=in_channels),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        return self.layer(x)

class OneNetv2(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, base_filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            PickLayer(),
            Conv2d1x1(in_channels * 4, base_filters),
            Conv2d1x1(base_filters, base_filters),
            PickLayer(),
            Conv2d1x1(base_filters * 4, base_filters * 4),
            Conv2d1x1(base_filters * 4, base_filters * 4),
            PickLayer(),
            Conv2d1x1(base_filters * 4 * 4, base_filters * 4 * 4),
            Conv2d1x1(base_filters * 4 * 4, base_filters * 4),
            PickLayer(),
            Conv2d1x1(base_filters * 4 * 4, base_filters * 4 * 4),
            Conv2d1x1(base_filters * 4 * 4, base_filters * 4 * 4),
            PickLayer(),
            Conv2d1x1(base_filters * 4 * 4 * 4, base_filters * 4 * 4 * 4),
            DWBlock(base_filters * 4 * 4 * 4, base_filters * 4 * 4 * 4),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_filters * 4 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class OneNetv3(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, base_filters: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            PickLayer(),
            PickLayer(),
            Conv2d1x1(in_channels * 4 * 4, base_filters),
            PickLayer(),
            PickLayer(),
            Conv2d1x1(base_filters * 4 * 4, base_filters * 4),
            PickLayer(),
            Conv2d1x1(base_filters * 4 * 4, base_filters * 4 * 4),
            Conv2d1x1(base_filters * 4 * 4, base_filters * 4 * 4),
            Conv2d1x1(base_filters * 4 * 4, base_filters * 4 * 4),
            Conv2d1x1(base_filters * 4 * 4, base_filters * 4 * 4),
            Conv2d1x1(base_filters * 4 * 4, base_filters * 4 * 4),
            Conv2d1x1(base_filters * 4 * 4, base_filters * 4 * 4),
            Conv2d1x1(base_filters * 4 * 4, base_filters * 4 * 4),
            Conv2d1x1(base_filters * 4 * 4, base_filters * 4 * 4),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_filters * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x