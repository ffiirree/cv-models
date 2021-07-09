import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF

__all__ = ['MultiInputModel']

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size:int = 3, stride: int = 1, padding: int = 1):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.layer(x)

class Pick(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f'{x.dim()} != 4'
        return torch.cat([x[:, :, 0::2, 0::2], 
                          x[:, :, 1::2, 0::2], 
                          x[:, :, 0::2, 1::2], 
                          x[:, :, 1::2, 1::2]],
                         dim=1
                        )

class MultiInputModel(nn.Module):
    def __init__(self, in_channels, num_classes, filters):
        super().__init__()

        self.s1 = nn.Sequential(
            Conv2dBlock(in_channels, filters *   1, 7, stride=2, padding=3),
            Conv2dBlock(filters *  1, filters *  1, 3),
            Conv2dBlock(filters *  1, filters *  1, 3),
            Conv2dBlock(filters *  1, filters *  2, 5, stride=2, padding=2),
            Conv2dBlock(filters *  2, filters *  2, 3),
            Conv2dBlock(filters *  2, filters *  2, 3),
        )

        self.s2 = nn.Sequential(
            Conv2dBlock(filters *  2, filters *  4, 5, stride=2, padding=2),
            Conv2dBlock(filters *  4, filters *  4, 3),
            Conv2dBlock(filters *  4, filters *  4, 3),
            Conv2dBlock(filters *  4, filters *  8, 5, stride=2, padding=2),
            Conv2dBlock(filters *  8, filters *  8, 3),
            Conv2dBlock(filters *  8, filters *  8, 3),
        )

        self.s3 = nn.Sequential(
            Conv2dBlock(filters *  8 + in_channels * 16, filters * 16, 5, stride=2, padding=2),
            Conv2dBlock(filters * 16, filters * 16, 3),
            Conv2dBlock(filters * 16, filters * 16, 3),
        )

        self.pick = Pick()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 16, num_classes)

    def forward(self, x):
        # identity = TF.resize(x, [112, 112])
        identity = TF.resize(x, [56, 56])
        
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(torch.cat([x, self.pick(self.pick(identity))], dim=1))
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x