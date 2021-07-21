import torch
import numpy as np

__all__ = ['MixUp']


class MixUp():
    def __init__(self, alpha, criterion):
        self.alpha = alpha
        self.lam = 1
        self.criterion = criterion

    def mix(self, x: torch.Tensor, y: torch.Tensor):
        self.lam = np.random.beta(self.alpha, self.alpha)
        indices = torch.randperm(x.shape[0]).cuda()
        x = self.lam * x + (1 - self.lam) * x[indices, :]
        y = [y, y[indices]]
        return x, y

    def loss(self, o, y):
        return self.lam * self.criterion(o, y[0]) + (1 - self.lam) * self.criterion(o, y[1])

    def __repr__(self) -> str:
        return f'MixUp(alpha={self.alpha})'
