import torch
import torch.nn as nn
from .utils import export, load_from_local_or_url
from typing import Any


@export
class AlexNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        dropout_rate: float = 0.5,
        thumbnail: bool = False,
        **kwargs: Any
    ):
        super().__init__()

        FRONT_S = 1 if thumbnail else 4

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=11,
                      stride=FRONT_S, padding=2),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.pool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


@export
def alexnet(pretrained: bool = False, pth: str = None, progress: bool = False, **kwargs: Any):
    model = AlexNet(**kwargs)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model
