from torch import nn
import torch.nn.functional as F
from typing import Optional, List


class Interpolate(nn.Module):
    def __init__(self,  mode='nearest') -> None:
        super().__init__()

        self.mode = mode

    def forward(self, x, size: Optional[int] = None, scale_factor: Optional[List[float]] = None):
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=self.mode)

    def extra_repr(self) -> str:
        return 'mode=' + self.mode
