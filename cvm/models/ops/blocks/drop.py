import torch
from torch import nn


class DropPath(nn.Module):
    """Stochastic Depth: Drop paths per sample (when applied in main path of residual blocks)"""

    def __init__(self, survival_prob: float):
        super().__init__()

        self.p = survival_prob

    def forward(self, x):
        if self.p == 1. or not self.training:
            return x

        # work with diff dim tensors, not just 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        probs = self.p + torch.rand(shape, dtype=x.dtype, device=x.device)
        # We therefore need to re-calibrate the outputs of any given function f
        # by the expected number of times it participates in training, p.
        return (x / self.p) * probs.floor_()

    def extra_repr(self):
        return f'survival_prob={self.p}'
