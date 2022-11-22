import torch
from torch import nn


class ChannelChunk(nn.Module):
    def __init__(self, groups: int):
        super().__init__()

        self.groups = groups

    def forward(self, x: torch.Tensor):
        return torch.chunk(x, self.groups, dim=1)

    def extra_repr(self):
        return f'groups={self.groups}'


class ChannelSplit(nn.Module):
    def __init__(self, sections):
        super().__init__()

        self.sections = sections

    def forward(self, x: torch.Tensor):
        return torch.split(x, self.sections, dim=1)

    def extra_repr(self):
        return f'sections={self.sections}'
