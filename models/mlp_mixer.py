import torch
import torch.nn as nn
from .core import blocks

__all__ = ['Mixer']

class Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        ...