import torch.nn as nn
from . import norm_act
from .vanilla_conv2d import Conv2d1x1Block, Conv2d1x1BN, Conv2dBlock
from .depthwise_separable_conv2d import DepthwiseBlock, DepthwiseConv2dBN
from .squeeze_excite import SEBlock
from .channel_combine import Combine
from .drop import DropPath


class InvertedResidualBlock(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        t,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        se_ratio: float = None,
        se_ind: bool = False,
        survival_prob: float = None,
        normalizer_fn: nn.Module = None,
        activation_fn: nn.Module = None,
        dw_se_act: nn.Module = None
    ):
        super().__init__()

        self.inp = inp
        self.planes = int(self.inp * t)
        self.oup = oup
        self.stride = stride
        self.apply_residual = (self.stride == 1) and (self.inp == self.oup)
        self.se_ratio = se_ratio if se_ind or se_ratio is None else (se_ratio / t)
        self.has_se = (self.se_ratio is not None) and (self.se_ratio > 0) and (self.se_ratio <= 1)

        normalizer_fn = normalizer_fn or norm_act._NORMALIZER
        activation_fn = activation_fn or norm_act._ACTIVATION

        layers = []
        if t != 1:
            layers.append(Conv2d1x1Block(inp, self.planes, normalizer_fn=normalizer_fn, activation_fn=activation_fn))

        if dw_se_act is None:
            layers.append(DepthwiseBlock(self.planes, self.planes, kernel_size, stride=self.stride,
                                         padding=padding, dilation=dilation, normalizer_fn=normalizer_fn, activation_fn=activation_fn))
        else:
            layers.append(DepthwiseConv2dBN(self.planes, self.planes, kernel_size, stride=self.stride, padding=padding,
                                            dilation=dilation, normalizer_fn=normalizer_fn))

        if self.has_se:
            layers.append(SEBlock(self.planes, self.se_ratio))

        if dw_se_act:
            layers.append(dw_se_act())

        layers.append(Conv2d1x1BN(self.planes, oup, normalizer_fn=normalizer_fn))

        if self.apply_residual and survival_prob:
            layers.append(DropPath(survival_prob))

        self.branch1 = nn.Sequential(*layers)
        self.branch2 = nn.Identity() if self.apply_residual else None
        self.combine = Combine('ADD') if self.apply_residual else None

    def forward(self, x):
        if self.apply_residual:
            return self.combine([self.branch2(x), self.branch1(x)])
        else:
            return self.branch1(x)


class FusedInvertedResidualBlock(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        t,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        se_ratio: float = None,
        se_ind: bool = False,
        survival_prob: float = None,
        normalizer_fn: nn.Module = None,
        activation_fn: nn.Module = None
    ):
        super().__init__()

        self.inp = inp
        self.planes = int(self.inp * t)
        self.oup = oup
        self.stride = stride
        self.padding = padding if padding is not None else (kernel_size // 2)
        self.apply_residual = (self.stride == 1) and (self.inp == self.oup)
        self.se_ratio = se_ratio if se_ind or se_ratio is None else (se_ratio / t)
        self.has_se = (self.se_ratio is not None) and (self.se_ratio > 0) and (self.se_ratio <= 1)

        normalizer_fn = normalizer_fn or norm_act._NORMALIZER
        activation_fn = activation_fn or norm_act._ACTIVATION

        layers = [
            Conv2dBlock(inp, self.planes, kernel_size, stride=self.stride, padding=self.padding,
                        normalizer_fn=normalizer_fn, activation_fn=activation_fn)
        ]

        if self.has_se:
            layers.append(SEBlock(self.planes, self.se_ratio))

        layers.append(Conv2d1x1BN(
            self.planes, oup, normalizer_fn=normalizer_fn))

        if self.apply_residual and survival_prob:
            layers.append(DropPath(survival_prob))

        self.branch1 = nn.Sequential(*layers)
        self.branch2 = nn.Identity() if self.apply_residual else None
        self.combine = Combine('ADD') if self.apply_residual else None

    def forward(self, x):
        if self.apply_residual:
            return self.combine([self.branch2(x), self.branch1(x)])
        else:
            return self.branch1(x)
