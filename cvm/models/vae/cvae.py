import torch
import torch.nn as nn

from ..utils import export, load_from_local_or_url
from typing import Any


@export
class ConditionalVAE(nn.Module):
    """
        Paper: [Learning Structured Output Representation using Deep Conditional Generative Models](https://papers.nips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html)
    """
    def __init__(
        self,
        image_size,
        nz: int = 100,
        **kwargs: Any
    ):
        super().__init__()

        self.image_size = image_size
        self.nz = nz

        self.embeds_en = nn.Embedding(10, 200)

        self.embeds_de = nn.Embedding(10, 10)

        # Q(z|X)
        self.encoder = nn.Sequential(
            nn.Linear(self.image_size ** 2 + 200, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, self.nz * 2)
        )

        # P(X|z)
        self.decoder = nn.Sequential(
            nn.Linear(self.nz + 10, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, self.image_size ** 2),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, image_size, image_size))
        )

    def sample_z(self, mu, logvar, c):
        eps = torch.randn_like(logvar)

        return torch.cat([mu + eps * torch.exp(0.5 * logvar), c], dim=1)

    def forward(self, x, c):
        x = torch.flatten(x, 1)

        x = torch.cat([x, self.embeds_en(c)], dim=1)

        mu, logvar = torch.chunk(self.encoder(x), 2, dim=1)

        z = self.sample_z(mu, logvar, self.embeds_de(c))

        x = self.decoder(z)
        return x, mu, logvar


@export
def cvae(
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    model = ConditionalVAE(**kwargs)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model
