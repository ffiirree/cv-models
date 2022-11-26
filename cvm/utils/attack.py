import abc
import torch
import torch.nn.functional as F

from typing import Callable

__all__ = ['FGSM', 'PGD']


class Attacker(abc.ABC):
    def __init__(self, model, epsilon: float = 0.03):
        super().__init__()

        self.model = model
        self.model.eval()

        self.epsilon = epsilon

    @staticmethod
    def prepare_input(x: torch.Tensor):
        x.requires_grad_()

        if x.grad is not None:
            x.grad.zero_()

    perturb: Callable


class FGSM(Attacker):
    def __init__(self, model, epsilon: float = 0.05):
        super().__init__(model, epsilon=epsilon)

    def perturb(self, x: torch.Tensor, y: torch.Tensor = None, targeted: bool = False):
        super().prepare_input(x)

        self.model.zero_grad()

        loss = F.cross_entropy(self.model(x), y)
        loss.backward()

        eta = self.epsilon * torch.sign(x.grad)

        if not targeted:
            x = (x + eta).detach()
        else:
            x = (x - eta).detach()

        return torch.clamp(x, -1.0, 1.0)

    def __repr__(self) -> str:
        return f'FGSM(eps={self.epsilon})'


class PGD(Attacker):
    def __init__(self, model, epsilon: float = 0.05, k: int = 7, alpha: float = 0.01):
        super().__init__(model, epsilon=epsilon)

        self.k = k
        self.alpha = alpha

    def perturb(self, x: torch.Tensor, y: torch.Tensor = None, targeted: bool = False):
        x_nat = x.detach().clone()

        for _ in range(self.k):
            super().prepare_input(x)

            self.model.zero_grad()

            loss = F.cross_entropy(self.model(x), y)
            loss.backward()

            eta = self.alpha * torch.sign(x.grad)

            if not targeted:
                x = (x + eta).detach()
            else:
                x = (x - eta).detach()

            x = torch.clamp(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = torch.clamp(x, -1.0, 1.0)

        return x

    def __repr__(self) -> str:
        return f'PGD(eps={self.epsilon}, steps={self.k}, alpha={self.alpha})'
