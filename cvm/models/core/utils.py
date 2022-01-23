import os
import sys
import functools
import torch
import torch.nn as nn
from . import blocks

__all__ = ['export', 'config', 'load_from_local_or_url',
           'get_out_channels']


def export(obj):
    if hasattr(sys.modules[obj.__module__], '__all__'):
        assert obj.__name__ not in sys.modules[
            obj.__module__].__all__, f'Duplicate name: {obj.__name__}'

        sys.modules[obj.__module__].__all__.append(obj.__name__)
    else:
        sys.modules[obj.__module__].__all__ = [obj.__name__]
    return obj


def config(url='', **settings):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            kwargs['url'] = url
            # kwargs['arch'] = func.__name__
            return func(*args, **{**settings, **kwargs})
        return wrapper

    return decorator


def load_from_local_or_url(model, pth=None, url=None, progress=True):
    assert pth is not None or url is not None, 'The "pth" and "url" can not both be None.'

    if pth is not None:
        state_dict = torch.load(os.path.expanduser(pth))
    else:
        state_dict = torch.hub.load_state_dict_from_url(url, progress=progress)

    model.load_state_dict(state_dict)


def get_out_channels(module: nn.Module):
    # block has out_channels
    if isinstance(module, blocks.Stage) and hasattr(module, 'out_channels'):
        return module.out_channels

    # or get channels of the last Conv2d
    out_channels = 0
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            out_channels = m.out_channels

    return out_channels
