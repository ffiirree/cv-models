import torch
import torch.nn as nn
from .core import blocks

__all__ = ['EfficientNetv2']

class EfficientNetv2(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        ...