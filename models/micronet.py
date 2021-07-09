import torch
import torch.nn as nn

__all__ = ['MicroNet', 'MicroNetv2', 'MicroNetv3', 'MicroNetv4']

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size:int = 3, stride: int = 1, padding: int = 1):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.layer(x)

class DWBlock(nn.Module):
    def __init__(self, in_channels, kernel_size:int = 3, stride: int = 1, padding: int = 1):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, bias=False, stride=stride, padding=padding, groups=in_channels),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels)
        )
    
    def forward(self, x):
        return self.layer(x)

class Conv1x1Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride: int = 1):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        return self.layer(x)


class DWBlock2(nn.Module):
    def __init__(self, in_channels, kernel_size:int = 3, stride: int = 1, padding: int = 1):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, bias=False, stride=stride, padding=padding, groups=in_channels),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(in_channels)
        )
    
    def forward(self, x):
        return self.layer(x)

class Conv1x1Block2(nn.Module):
    def __init__(self, in_channels, out_channels, stride: int = 1):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        return self.layer(x)


class MicroNet(nn.Module):
    def __init__(self, in_channels:int=3, num_classes:int=1000, filters:int=32):
        super().__init__()

        self.dwblock16 = DWBlock(filters * 16)

        self.features = nn.Sequential(
            Conv2dBlock(in_channels, filters * 1, stride=2),

            DWBlock(filters * 1),
            Conv1x1Block(filters * 1, filters * 2),

            DWBlock(filters * 2, stride=2),
            Conv1x1Block(filters * 2, filters * 4),

            DWBlock(filters * 4),
            Conv1x1Block(filters * 4, filters * 4),

            DWBlock(filters * 4, stride=2),
            Conv1x1Block(filters * 4, filters * 8),

            DWBlock(filters * 8),
            Conv1x1Block(filters * 8, filters * 8),

            DWBlock(filters * 8, stride=2),
            Conv1x1Block(filters * 8, filters * 16),

            self.dwblock16,
            Conv1x1Block(filters * 16, filters * 16),

            self.dwblock16,
            Conv1x1Block(filters * 16, filters * 16),

            self.dwblock16,
            Conv1x1Block(filters * 16, filters * 16),

            self.dwblock16,
            Conv1x1Block(filters * 16, filters * 16),

            self.dwblock16,
            Conv1x1Block(filters * 16, filters * 16),

            self.dwblock16,
            Conv1x1Block(filters * 16, filters * 16),

            self.dwblock16,
            Conv1x1Block(filters * 16, filters * 16),

            self.dwblock16,
            Conv1x1Block(filters * 16, filters * 16),

            self.dwblock16,
            Conv1x1Block(filters * 16, filters * 16),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MicroNetv2(nn.Module):
    def __init__(self, in_channels:int=3, num_classes:int=1000, filters:int=32):
        super().__init__()

        # self.dwblock16 = DWBlock(filters * 16)
        self.conv1x1 = Conv1x1Block(filters * 16, filters * 16)

        self.features = nn.Sequential(
            Conv2dBlock(in_channels, filters * 1, stride=2),

            DWBlock(filters * 1),
            Conv1x1Block(filters * 1, filters * 2),

            DWBlock(filters * 2, stride=2),
            Conv1x1Block(filters * 2, filters * 4),

            DWBlock(filters * 4),
            Conv1x1Block(filters * 4, filters * 4),

            DWBlock(filters * 4, stride=2),
            Conv1x1Block(filters * 4, filters * 8),

            DWBlock(filters * 8),
            Conv1x1Block(filters * 8, filters * 8),

            DWBlock(filters * 8, stride=2),
            Conv1x1Block(filters * 8, filters * 16),

            DWBlock(filters * 16),
            self.conv1x1,

            DWBlock(filters * 16),
            self.conv1x1,

            DWBlock(filters * 16),
            self.conv1x1,

            DWBlock(filters * 16),
            self.conv1x1,

            DWBlock(filters * 16),
            self.conv1x1,

            DWBlock(filters * 16, stride=2),
            Conv1x1Block(filters * 16, filters * 32),

            DWBlock(filters * 32),
            Conv1x1Block(filters * 32, filters * 32),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class MicroNetv3(nn.Module):
    def __init__(self, in_channels:int=3, num_classes:int=1000, filters:int=32):
        super().__init__()

        self.dwblock16 = DWBlock2(filters * 16)
        self.conv1x1 = Conv1x1Block(filters * 16, filters * 16)

        self.features = nn.Sequential(
            Conv2dBlock(in_channels, filters * 1, stride=2),

            DWBlock(filters * 1),
            Conv1x1Block(filters * 1, filters * 2),

            DWBlock(filters * 2, stride=2),
            Conv1x1Block(filters * 2, filters * 4),

            DWBlock(filters * 4),
            Conv1x1Block(filters * 4, filters * 4),

            DWBlock(filters * 4, stride=2),
            Conv1x1Block(filters * 4, filters * 8),

            DWBlock(filters * 8),
            Conv1x1Block(filters * 8, filters * 8),

            DWBlock(filters * 8, stride=2),
            Conv1x1Block(filters * 8, filters * 16),

            self.dwblock16,
            self.conv1x1,

            self.dwblock16,
            self.conv1x1,

            self.dwblock16,
            self.conv1x1,

            self.dwblock16,
            self.conv1x1,

            self.dwblock16,
            self.conv1x1,

            DWBlock(filters * 16, stride=2),
            Conv1x1Block(filters * 16, filters * 32),

            DWBlock(filters * 32),
            Conv1x1Block(filters * 32, filters * 32),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MicroNetv4(nn.Module):
    def __init__(self, in_channels:int=3, num_classes:int=1000, filters:int=32):
        super().__init__()

        self.dwblock16 = DWBlock(filters * 16)
        self.conv1x1 = Conv1x1Block(filters * 16, filters * 16)

        self.features = nn.Sequential(
            Conv2dBlock(in_channels, filters * 1, stride=2),

            DWBlock(filters * 1),
            Conv1x1Block(filters * 1, filters * 2),

            DWBlock(filters * 2, stride=2),
            Conv1x1Block(filters * 2, filters * 4),

            DWBlock(filters * 4),
            Conv1x1Block(filters * 4, filters * 4),

            DWBlock(filters * 4, stride=2),
            Conv1x1Block(filters * 4, filters * 8),

            DWBlock(filters * 8),
            Conv1x1Block(filters * 8, filters * 8),

            DWBlock(filters * 8, stride=2),
            Conv1x1Block(filters * 8, filters * 16),

            self.dwblock16,
            self.conv1x1,

            DWBlock(filters * 16, stride=2),
            Conv1x1Block(filters * 16, filters * 32),

            DWBlock(filters * 32),
            Conv1x1Block(filters * 32, filters * 32),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MicroNetv5(nn.Module):
    def __init__(self, in_channels:int=3, num_classes:int=1000, filters:int=32):
        super().__init__()

        self.dwblock16 = DWBlock(filters * 16)

        self.features = nn.Sequential(
            Conv2dBlock(in_channels, filters * 1, stride=2),

            DWBlock(filters * 1),
            Conv1x1Block(filters * 1, filters * 2),

            DWBlock(filters * 2, stride=2),
            Conv1x1Block(filters * 2, filters * 4),

            DWBlock(filters * 4),
            Conv1x1Block(filters * 4, filters * 4),

            DWBlock(filters * 4, stride=2),
            Conv1x1Block(filters * 4, filters * 8),

            DWBlock(filters * 8),
            Conv1x1Block(filters * 8, filters * 8),

            DWBlock(filters * 8, stride=2),
            Conv1x1Block(filters * 8, filters * 16),

            self.dwblock16,
            Conv1x1Block(filters * 16, filters * 16),

            self.dwblock16,
            Conv1x1Block(filters * 16, filters * 16),

            self.dwblock16,
            Conv1x1Block(filters * 16, filters * 16),

            self.dwblock16,
            Conv1x1Block(filters * 16, filters * 16),

            self.dwblock16,
            Conv1x1Block(filters * 16, filters * 16),

            DWBlock(filters * 16, stride=2),
            Conv1x1Block(filters * 16, filters * 32),

            DWBlock(filters * 32),
            Conv1x1Block(filters * 32, filters * 32),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

