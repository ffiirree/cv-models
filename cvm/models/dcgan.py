import torch.nn as nn
from .core import blocks, export, load_from_local_or_url
from typing import Any


@export
class DCGAN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 100,
        in_channels: int = 3,
        **kwargs: Any
    ) -> None:
        super().__init__()
        
        base_width = 64

        self.generator = nn.Sequential(
            # input : (batch_size, hidden_dim, 1, 1)
            nn.ConvTranspose2d(hidden_dim, base_width * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(True),
            # state size : (batch_size, ngf * 8, 4, 4)
            nn.ConvTranspose2d(base_width * 8, base_width * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(True),
            # state size : (batch_size, ngf * 4, 8, 8)
            nn.ConvTranspose2d(base_width * 4, base_width * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(True),
            # state size: (batch_size, ngf * 2, 16, 16)
            nn.ConvTranspose2d(base_width * 2, base_width, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(True),
            # state size : (batch_size, ngf, 32, 32)
            nn.ConvTranspose2d(base_width, in_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size : (batch_size, nc, 64, 64)
        )

        self.discriminator = nn.Sequential(
            # input size : (batch_size, nc, 64, 64)
            nn.Conv2d(in_channels, base_width, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size : (batch_size, base_width, 32, 32)
            nn.Conv2d(base_width, base_width * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_width * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #state size : (batch_size, base_width * 2, 16, 16)
            nn.Conv2d(base_width * 2, base_width * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_width * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size : (batch_size, base_width * 4, 8, 8)
            nn.Conv2d(base_width * 4, base_width * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_width * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size : (batch_size, base_width * 8, 4, 4)
            nn.Conv2d(base_width * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # state size : (batch_size, 1, 1, 1)
            nn.Flatten()
        )


@export
def dcgan(
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    model = DCGAN(**kwargs)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model
