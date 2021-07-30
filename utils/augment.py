import torch
import numpy as np

__all__ = ['MixUp']


class MixUp(object):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.lam = 1

    def mix(self, x: torch.Tensor, y: torch.Tensor):
        self.lam = np.random.beta(self.alpha, self.alpha)
        indices = torch.randperm(x.shape[0]).to(x.device)
        x = self.lam * x + (1 - self.lam) * x[indices, :]
        y = self.lam * y + (1 - self.lam) * y[indices, :]
        return x, y

    def __repr__(self) -> str:
        return f'MixUp(alpha={self.alpha})'


def augment_list():
    return []


class RandAugment:
    def __init__(self, N, M):
        self.n = N
        self.m = M

        self.augment_list = augment_list()

    def __call__(self, image):
        pass
