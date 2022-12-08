import torch
import torch.nn as nn

from ..utils import export, load_from_local_or_url
from typing import Any


@export
class VAE(nn.Module):
    def __init__(
        self,
        image_size,
        nz: int = 100,
        **kwargs: Any
    ):
        super().__init__()

        self.image_size = image_size
        self.nz = nz

        # Q(z|X)
        self.encoder = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.image_size ** 2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, self.nz * 2)
        )

        # P(X|z)
        self.decoder = nn.Sequential(
            nn.Linear(self.nz, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, self.image_size ** 2),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, image_size, image_size))
        )

    def sample_z(self, mu, logvar):
        eps = torch.randn_like(logvar)

        return mu + eps * torch.exp(0.5 * logvar)

    def forward(self, x):
        mu, logvar = torch.chunk(self.encoder(x), 2, dim=1)

        z = self.sample_z(mu, logvar)

        x = self.decoder(z)
        return x, mu, logvar


@export
def vae(
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    model = VAE(**kwargs)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model
