from functools import partial
import pytest
import torch
import torch.nn as nn
from cvm.models.ops import blocks


def test_se_block_forward():
    inputs = torch.randn(16, 3, 56, 56)

    se = blocks.SEBlock(3, 0.25)

    outputs = se(inputs)
    assert outputs.shape == inputs.shape
    assert isinstance(se.act, nn.ReLU)
    assert isinstance(se.gate, nn.Sigmoid)


def test_se_block_decorator():
    with blocks.se(inner_nonlinear=nn.SiLU, gating_fn=nn.Hardsigmoid):
        se = blocks.SEBlock(3, 0.25)

    assert isinstance(se.act, nn.SiLU)
    assert isinstance(se.gate, nn.Hardsigmoid)


def test_normalizer_decorator():
    with blocks.normalizer(None):
        layers = blocks.norm_activation(3)

    assert len(layers) == 1
    assert isinstance(layers[0], nn.ReLU)

    with blocks.normalizer(nn.LayerNorm, position='before'):
        layers = blocks.norm_activation(3)

    assert len(layers) == 2
    assert isinstance(layers[0], nn.LayerNorm)
    assert isinstance(layers[1], nn.ReLU)

    with blocks.normalizer(partial(nn.BatchNorm2d, eps=0.1), position='after'):
        layers = blocks.norm_activation(3)

    assert len(layers) == 2
    assert isinstance(layers[0], nn.ReLU)
    assert isinstance(layers[1], nn.BatchNorm2d)
    assert layers[1].eps == 0.1


def test_nonlinear_decorator():
    with blocks.nonlinear(None):
        layers = blocks.norm_activation(3)

    assert len(layers) == 1
    assert isinstance(layers[0], nn.BatchNorm2d)

    with blocks.nonlinear(nn.SiLU):
        layers = blocks.norm_activation(3)

    assert len(layers) == 2
    assert isinstance(layers[0], nn.BatchNorm2d)
    assert isinstance(layers[1], nn.SiLU)
