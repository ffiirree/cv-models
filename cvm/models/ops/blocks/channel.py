import torch
from torch import nn
from ..functional import channel_shuffle
from typing import List


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


class ChannelShuffle(nn.Module):
    def __init__(self, groups: int):
        super().__init__()

        self.groups = groups

    def forward(self, x):
        return channel_shuffle(x, self.groups)

    def extra_repr(self):
        return 'groups={}'.format(self.groups)


class Combine(nn.Module):
    def __init__(self, method: str = 'ADD', *args, **kwargs):
        super().__init__()
        assert method in ['ADD', 'CONCAT'], ''

        self.method = method
        self._combine = self._add if self.method == 'ADD' else self._cat

    @staticmethod
    def _add(x):
        return x[0] + x[1]

    @staticmethod
    def _cat(x):
        return torch.cat(x, dim=1)

    def forward(self, x):
        return self._combine(x)

    def extra_repr(self):
        return f'method=\'{self.method}\''


class ConcatBranches(nn.Sequential):
    def forward(self, x):
        res = []
        for module in self:
            res.append(module(x))
        return torch.cat(res, dim=1)


class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

    def extra_repr(self):
        return ', '.join([str(dim) for dim in self.dims])
