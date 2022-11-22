from torch import nn
from typing import Union, List


class Stage(nn.Sequential):
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], list):
            args = args[0]
        super().__init__(*args)

    def append(self, m: Union[nn.Module, List[nn.Module]]):
        if isinstance(m, nn.Module):
            self.add_module(str(len(self)), m)
        elif isinstance(m, list):
            [self.append(i) for i in m]
        else:
            ValueError('')
