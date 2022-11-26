import abc
import torch

from typing import Callable


class Attacker(abc.ABC):
    def __init__(self, model, epsilon: float = 0.03, mean=None, std=None):
        super().__init__()

        self.model = model
        self.model.eval()

        self.epsilon = epsilon

        self.mean = None
        self.std = None

        self.normalized = None  # None, False, True

    def set_nomarlized(self, mean, std):
        self.mean = mean
        self.std = std

        self.normalized = True

    def normalize(self, x: torch.Tensor):
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)

        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)

        return (x - mean) / std

    def inverse_normalize(self, x: torch.Tensor):
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)

        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)

        return x * std + mean

    def prepare_inputs(self, x):
        if self.normalized is True:
            x = self.inverse_normalize(x)
            self.normalized = False

        x.requires_grad_(True)
        return x

    def unprepare_inputs(self, x):
        if self.normalized is False:
            x = self.normalize(x)
            self.normalized = True

        return x

    def forward(self, x):
        if self.normalized is False:
            x = self.normalize(x)

        return self.model(x)

    perturb: Callable
