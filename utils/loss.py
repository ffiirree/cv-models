import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LabelSmoothingCrossEntropyLoss', 'SoftLabelCrossEntropyLoss']


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


class SoftLabelCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        logprobs = F.log_softmax(x, dim=-1)
        loss = -(logprobs * y).sum(dim=-1)
        return loss.mean()
