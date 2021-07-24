import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['MixUp', 'LabelSmoothingCrossEntropyLoss']


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


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.confidence = 1 - smoothing
        self.smoothing = smoothing

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        logprobs = F.log_softmax(x, dim=-1)

        nll_loss = -(logprobs * y).sum(dim=-1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        return loss.mean()
