from typing import List
from contextlib import contextmanager
from functools import partial
import torch.nn as nn

_NORM_POSIITON: str = 'before'
_NORMALIZER: nn.Module = nn.BatchNorm2d
_ACTIVATION: nn.Module = partial(nn.ReLU, inplace=True)


class Nil:
    ...


@contextmanager
def normalizer(
    # _NORMALIZER can be None, Nil: _NORMALIZER->_NORMALIZER, None: _NORMALIZER->None
    fn: nn.Module = Nil,
    position: str = None
):

    global _NORMALIZER, _NORM_POSIITON

    fn = _NORMALIZER if fn == Nil else fn
    position = position or _NORM_POSIITON

    _pre_normalizer = _NORMALIZER
    _pre_position = _NORM_POSIITON

    _NORMALIZER = fn
    _NORM_POSIITON = position

    yield

    _NORMALIZER = _pre_normalizer
    _NORM_POSIITON = _pre_position


@contextmanager
def activation(fn: nn.Module):
    global _ACTIVATION

    _pre_activation = _ACTIVATION
    _ACTIVATION = fn
    yield
    _ACTIVATION = _pre_activation


def normalizer_fn(channels):
    return _NORMALIZER(channels)


def activation_fn():
    return _ACTIVATION()


def norm_activation(
    channels,
    normalizer_fn: nn.Module = None,
    activation_fn: nn.Module = None,
    norm_position: str = None
) -> List[nn.Module]:
    norm_position = norm_position or _NORM_POSIITON
    assert norm_position in ['before', 'after', 'none'], ''

    normalizer_fn = normalizer_fn or _NORMALIZER
    activation_fn = activation_fn or _ACTIVATION

    if normalizer_fn == None and activation_fn == None:
        return []

    if normalizer_fn == None:
        return [activation_fn()]

    if activation_fn == None:
        return [normalizer_fn(channels)]

    if norm_position == 'after':
        return [activation_fn(), normalizer_fn(channels)]

    return [normalizer_fn(channels), activation_fn()]
