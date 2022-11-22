import torch.nn as nn

from ..ops import blocks
from ..utils import export, load_from_local_or_url
from typing import Any


@export
class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        filters: int = [64, 128, 256, 512, 1024],
        **kwargs: Any
    ):
        super().__init__()

        for i in range(4):
            self.add_module(f'encode_conv{i+1}', nn.Sequential(
                blocks.Conv2dBlock(filters[i - 1] if i else in_channels, filters[i]),
                blocks.Conv2dBlock(filters[i], filters[i])
            ))
            self.add_module(f'down{i+1}', nn.MaxPool2d(2, 2))

        self.u = nn.Sequential(
            blocks.Conv2dBlock(filters[3], filters[4]),
            blocks.Conv2dBlock(filters[4], filters[4])
        )

        filters.reverse()
        for i in range(4):
            self.add_module(f'up{i+1}', nn.ConvTranspose2d(filters[i], filters[i + 1], 4, stride=2, padding=1))
            self.add_module(f'decode_conv{i+1}', nn.Sequential(
                blocks.Combine('CONCAT'),
                blocks.Conv2dBlock(filters[i], filters[i+1]),
                blocks.Conv2dBlock(filters[i + 1], filters[i + 1])
            ))

        self.output = blocks.Conv2d1x1(filters[-1], num_classes, bias=True)

    def forward(self, x):
        e1 = self.encode_conv1(x)
        e2 = self.encode_conv2(self.down1(e1))
        e3 = self.encode_conv3(self.down2(e2))
        e4 = self.encode_conv4(self.down3(e3))

        u = self.u(self.down4(e4))

        d1 = self.decode_conv1([e4, self.up1(u)])
        d2 = self.decode_conv2([e3, self.up2(d1)])
        d3 = self.decode_conv3([e2, self.up3(d2)])
        d4 = self.decode_conv4([e1, self.up4(d3)])

        return self.output(d4)


@export
def unet(
    pretrained: bool = False,
    pth: str = None,
    progress: bool = True,
    **kwargs: Any
):
    model = UNet(**kwargs)

    if pretrained:
        load_from_local_or_url(model, pth, kwargs.get('url', None), progress)
    return model
