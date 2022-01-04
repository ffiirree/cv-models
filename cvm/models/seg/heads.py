import torch.nn as nn


class FCNHead(nn.Sequential):
    def __init__(
        self,
        in_channels: int = 2048,
        channels: int = None,
        num_classes: int = 32,
        dropout_rate: float = 0.1,
    ):
        channels = channels or int(in_channels / 4.0)
        super().__init__(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(channels, num_classes, 1)
        )


class ClsHead(nn.Sequential):
    def __init__(
        self,
        in_channels,
        num_classes: int
    ):
        super().__init__(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(in_channels, num_classes)
        )
