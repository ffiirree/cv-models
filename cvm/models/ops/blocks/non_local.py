import torch
from torch import nn
from .vanilla_conv2d import Conv2d1x1
from ..functional import make_divisible


class NonLocalBlock(nn.Module):
    r"""Non-Local Block for image classification
    Paper: Non-local Neural Networks, https://arxiv.org/abs/1711.07971
    Code: https://github.com/facebookresearch/video-nonlocal-net
    """

    def __init__(
        self,
        in_channels,
        rd_ratio,
        rd_divisor: int = 8,
        use_scale: bool = True,
        use_norm: bool = True
    ):
        super().__init__()

        channels = make_divisible(in_channels * rd_ratio, rd_divisor)

        self.ratio = rd_ratio
        self.scale = channels ** -0.5 if use_scale else 1.0
        self.use_scale = use_scale

        # theta, phi, g
        self.W = Conv2d1x1(in_channels, channels * 3, bias=True)

        # z
        self.Z = nn.Sequential(
            Conv2d1x1(channels, in_channels, bias=not use_norm),
            nn.BatchNorm2d(in_channels) if use_norm else nn.Identity()
        )

        self.reset_parameters()

    def reset_parameters(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if len(list(m.parameters())) > 1:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 0.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, _, H, W = x.size()

        # self-attention: y = softmax((Q(x) @ K(x)) / N) @ V(x). @{
        t, p, g = torch.chunk(torch.flatten(self.W(x), start_dim=2), 3, dim=1)  # Q, K, V

        s = torch.einsum('ncq, nck -> nqk', t, p)
        s = torch.softmax(s * self.scale, dim=2)
        s = torch.einsum('nqv, ncv -> ncq', s, g)
        # @}

        z = self.Z(s.contiguous().view(N, -1, H, W))

        return z + x
