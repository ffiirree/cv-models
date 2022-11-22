import torch
from torch import nn 


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        self.alpha = nn.Parameter(torch.ones(dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(dim, 1, 1))

    def forward(self, x):
        return self.alpha * x + self.beta

    def extra_repr(self):
        return f'{self.dim}'