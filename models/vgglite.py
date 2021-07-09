import torch
import torch.nn as nn

__all__ = ['VGG13Lite']

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size:int = 3, stride: int = 1, padding: int = 1):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)

class VGG13Lite(nn.Module):
    def __init__(self, in_channels:int=3, num_classes:int=1000, filters:int=32):
        super().__init__()

        self.features = nn.Sequential(
            Conv2dBlock(in_channels, filters *   1, 3),
            Conv2dBlock(filters *  1, filters *  1, 3),
            nn.MaxPool2d(),

            Conv2dBlock(filters *  1, filters *  2, 3),
            Conv2dBlock(filters *  2, filters *  2, 3),
            nn.MaxPool2d(),

            Conv2dBlock(filters *  2, filters *  4, 3),
            Conv2dBlock(filters *  4, filters *  4, 3),
            nn.MaxPool2d(),

            Conv2dBlock(filters *  4, filters *  8, 3),
            Conv2dBlock(filters *  8, filters *  8, 3),
            nn.MaxPool2d(),

            Conv2dBlock(filters *  8, filters *  16, 3),
            Conv2dBlock(filters *  16, filters * 16, 3),
            nn.MaxPool2d(),

            Conv2dBlock(filters *  16, filters * 16, 3),
            Conv2dBlock(filters *  16, filters * 16, 3),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x