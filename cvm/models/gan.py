import torch
import torch.nn as nn
from .core import blocks, export, load_from_local_or_url
from typing import Any


@export
class GAN(nn.Module):
    def __init__(
        self,
        nz: int = 100,
        image_size: int = 28,
        **kwargs: Any
    ) -> None:
        super().__init__()

        self.generator = nn.Sequential(
            nn.Linear(nz, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, image_size ** 2),
            nn.Tanh()  # Training failed when using the 'sigmoid'
        )

        self.discriminator = nn.Sequential(
            nn.Linear(image_size * image_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )


@export
def gan(
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    model = GAN(**kwargs)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model
