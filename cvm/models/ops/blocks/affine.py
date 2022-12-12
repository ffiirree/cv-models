import torch
from torch import nn


class Scale(nn.Module):
    def __init__(self, dim, alpha: float = 1e-6):
        super().__init__()

        self.dim = dim

        self.alpha = nn.Parameter(torch.ones(dim, 1, 1).fill_(alpha))

    def forward(self, x):
        return self.alpha * x

    def extra_repr(self):
        return f'{self.dim}'


class Affine(nn.Module):
    def __init__(self, dim, alpha: float = 1.0, beta: float = 0.0):
        super().__init__()

        self.dim = dim

        self.alpha = nn.Parameter(torch.empty(dim, 1, 1).fill_(alpha))
        self.beta = nn.Parameter(torch.empty(dim, 1, 1).fill_(beta))

    def forward(self, x):
        return self.alpha * x + self.beta

    def extra_repr(self):
        return f'{self.dim}'
