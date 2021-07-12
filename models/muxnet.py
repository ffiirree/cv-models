import torch
import torch.nn as nn

__all__ = ['MUXNet', 'MUXNetv2', 'MUXNetv3']


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      bias=False, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.layer(x)


class DWBlock(nn.Module):
    def __init__(self, channels, kernel_size: int = 3, stride: int = 1, padding: int = 1, mux_layer: nn.Module = None):
        super().__init__()

        self.channels = channels
        self.mux_layer = mux_layer

        if self.mux_layer is not None:
            self.channels = channels // 2

        self.layer = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=kernel_size,
                      bias=False, stride=stride, padding=padding, groups=self.channels),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.channels),
        )

    def forward(self, x):
        if self.mux_layer is not None:
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1 = self.layer(x1)
            x2 = self.mux_layer(x2)
            return torch.cat([x1, x2], dim=1)
        else:
            return self.layer(x)


class PWBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride: int = 1):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=1, bias=False, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.layer(x)


class MUXNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.in_channels = in_channels
        self.filters = filters

        self.features = self.make_layers()

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 32, num_classes)

    def make_layers(self):
        dw1 = DWBlock(self.filters * 1)

        dw2 = DWBlock(self.filters * 2)
        dw2s2 = DWBlock(self.filters * 2, stride=2)

        dw4 = DWBlock(self.filters * 4, mux_layer=dw2)
        dw4s2 = DWBlock(self.filters * 4, stride=2, mux_layer=dw2s2)

        dw8 = DWBlock(self.filters * 8, mux_layer=dw4)
        dw8s2 = DWBlock(self.filters * 8, stride=2, mux_layer=dw4s2)

        dw16 = DWBlock(self.filters * 16, mux_layer=dw8)
        dw16s2 = DWBlock(self.filters * 16, stride=2, mux_layer=dw8s2)

        dw32 = DWBlock(self.filters * 32, mux_layer=dw16)

        return nn.Sequential(
            Conv2dBlock(self.in_channels, self.filters * 1, stride=2),

            dw1,
            PWBlock(self.filters * 1, self.filters * 2),

            dw2s2,
            PWBlock(self.filters * 2, self.filters * 4),

            dw4,
            PWBlock(self.filters * 4, self.filters * 4),

            dw4s2,
            PWBlock(self.filters * 4, self.filters * 8),

            dw8,
            PWBlock(self.filters * 8, self.filters * 8),

            dw8s2,
            PWBlock(self.filters * 8, self.filters * 16),

            DWBlock(self.filters * 16, mux_layer=dw8),
            PWBlock(self.filters * 16, self.filters * 16),

            DWBlock(self.filters * 16, mux_layer=dw8),
            PWBlock(self.filters * 16, self.filters * 16),

            DWBlock(self.filters * 16, mux_layer=dw8),
            PWBlock(self.filters * 16, self.filters * 16),

            DWBlock(self.filters * 16, mux_layer=dw8),
            PWBlock(self.filters * 16, self.filters * 16),

            dw16,
            PWBlock(self.filters * 16, self.filters * 16),

            dw16s2,
            PWBlock(self.filters * 16, self.filters * 32),

            dw32,
            PWBlock(self.filters * 32, self.filters * 32),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class DWBlockv2(nn.Module):
    def __init__(self, channels, kernel_size: int = 3, stride: int = 1, padding: int = 1, mux_layer: nn.Module = None):
        super().__init__()

        self.channels = channels
        self.mux_layer = mux_layer

        if self.mux_layer is not None:
            self.channels = channels // 2

        self.layer = nn.Conv2d(self.channels, self.channels, kernel_size=kernel_size,
                               bias=False, stride=stride, padding=padding, groups=self.channels)

    def forward(self, x):
        if self.mux_layer is not None:
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1 = self.layer(x1)
            x2 = self.mux_layer(x2)
            return torch.cat([x1, x2], dim=1)
        else:
            return self.layer(x)


class MUXNetv2(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.in_channels = in_channels
        self.filters = filters

        self.features = self.make_layers()

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 32, num_classes)

    def make_layers(self):
        dw1 = DWBlockv2(self.filters * 1)

        dw2 = DWBlockv2(self.filters * 2)
        dw2s2 = DWBlockv2(self.filters * 2, stride=2)

        dw4 = DWBlockv2(self.filters * 4, mux_layer=dw2)
        dw4s2 = DWBlockv2(self.filters * 4, stride=2, mux_layer=dw2s2)

        dw8 = DWBlockv2(self.filters * 8, mux_layer=dw4)
        dw8s2 = DWBlockv2(self.filters * 8, stride=2, mux_layer=dw4s2)

        dw16 = DWBlockv2(self.filters * 16, mux_layer=dw8)
        dw16s2 = DWBlockv2(self.filters * 16, stride=2, mux_layer=dw8s2)

        dw32 = DWBlockv2(self.filters * 32, mux_layer=dw16)

        return nn.Sequential(
            Conv2dBlock(self.in_channels, self.filters * 1, stride=2),

            dw1,
            PWBlock(self.filters * 1, self.filters * 2),

            dw2s2,
            PWBlock(self.filters * 2, self.filters * 4),

            dw4,
            PWBlock(self.filters * 4, self.filters * 4),

            dw4s2,
            PWBlock(self.filters * 4, self.filters * 8),

            dw8,
            PWBlock(self.filters * 8, self.filters * 8),

            dw8s2,
            PWBlock(self.filters * 8, self.filters * 16),

            DWBlockv2(self.filters * 16, mux_layer=dw8),
            PWBlock(self.filters * 16, self.filters * 16),

            DWBlockv2(self.filters * 16, mux_layer=dw8),
            PWBlock(self.filters * 16, self.filters * 16),

            DWBlockv2(self.filters * 16, mux_layer=dw8),
            PWBlock(self.filters * 16, self.filters * 16),

            DWBlockv2(self.filters * 16, mux_layer=dw8),
            PWBlock(self.filters * 16, self.filters * 16),

            dw16,
            PWBlock(self.filters * 16, self.filters * 16),

            dw16s2,
            PWBlock(self.filters * 16, self.filters * 32),

            dw32,
            PWBlock(self.filters * 32, self.filters * 32),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class NonLinearBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return self.layer(x)

class MUXNetv3(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, filters: int = 32):
        super().__init__()

        self.in_channels = in_channels
        self.filters = filters

        self.features = self.make_layers()

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 32, num_classes)

    def make_layers(self):
        dw1 = DWBlockv2(self.filters * 1)

        dw2 = DWBlockv2(self.filters * 2)
        dw2s2 = DWBlockv2(self.filters * 2, stride=2)

        dw4 = DWBlockv2(self.filters * 4, mux_layer=dw2)
        dw4s2 = DWBlockv2(self.filters * 4, stride=2, mux_layer=dw2s2)

        dw8 = DWBlockv2(self.filters * 8, mux_layer=dw4)
        dw8s2 = DWBlockv2(self.filters * 8, stride=2, mux_layer=dw4s2)

        dw16 = DWBlockv2(self.filters * 16, mux_layer=dw8)
        dw16s2 = DWBlockv2(self.filters * 16, stride=2, mux_layer=dw8s2)

        dw32 = DWBlockv2(self.filters * 32, mux_layer=dw16)

        return nn.Sequential(
            Conv2dBlock(self.in_channels, self.filters * 1, stride=2),

            dw1,
            NonLinearBlock(self.filters),
            PWBlock(self.filters * 1, self.filters * 2),

            dw2s2,
            NonLinearBlock(self.filters * 2),
            PWBlock(self.filters * 2, self.filters * 4),

            dw4,
            NonLinearBlock(self.filters * 4),
            PWBlock(self.filters * 4, self.filters * 4),

            dw4s2,
            NonLinearBlock(self.filters * 4),
            PWBlock(self.filters * 4, self.filters * 8),

            dw8,
            NonLinearBlock(self.filters * 8),
            PWBlock(self.filters * 8, self.filters * 8),

            dw8s2,
            NonLinearBlock(self.filters * 8),
            PWBlock(self.filters * 8, self.filters * 16),

            DWBlockv2(self.filters * 16, mux_layer=dw8),
            NonLinearBlock(self.filters * 16),
            PWBlock(self.filters * 16, self.filters * 16),

            DWBlockv2(self.filters * 16, mux_layer=dw8),
            NonLinearBlock(self.filters * 16),
            PWBlock(self.filters * 16, self.filters * 16),

            DWBlockv2(self.filters * 16, mux_layer=dw8),
            NonLinearBlock(self.filters * 16),
            PWBlock(self.filters * 16, self.filters * 16),

            DWBlockv2(self.filters * 16, mux_layer=dw8),
            NonLinearBlock(self.filters * 16),
            PWBlock(self.filters * 16, self.filters * 16),

            dw16,
            NonLinearBlock(self.filters * 16),
            PWBlock(self.filters * 16, self.filters * 16),

            dw16s2,
            NonLinearBlock(self.filters * 16),
            PWBlock(self.filters * 16, self.filters * 32),

            dw32,
            NonLinearBlock(self.filters * 32),
            PWBlock(self.filters * 32, self.filters * 32),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
