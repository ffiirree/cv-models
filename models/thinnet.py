import torch
import torch.nn as nn

__all__ = ['ThinNet']

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 0, bn: bool = True):
        super(ConvBlock, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(inplace=True),
        ]
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        
        self.layer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layer(x)

class DuplicateConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 0, n_layers: int = 1, bn: bool = True):
        super(DuplicateConvBlock, self).__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.n_layers = n_layers
        self.bn = bn
        
        self.layer = self._make_layers()

    def forward(self, x: torch.Tensor):
        return self.layer(x)
    
    def _make_layers(self):
        conv2d = nn.Conv2d(self.channels, self.channels, self.kernel_size, stride=self.stride, padding=self.padding, bias=False)
        
        layers = []
        for _ in range(self.n_layers):
            layers.append(conv2d)
            layers.append(nn.ReLU(inplace=True)),
            if self.bn:
                layers.append(nn.BatchNorm2d(self.channels))
        
        return nn.Sequential(*layers)

class ThinNet(nn.Module):
    def __init__(self, in_channels: int = 1, n_classes: int = 10, filters: list = [32, 64, 128], n_blocks: list = [1, 1, 1], n_layers: int = [1, 1, 1], bn: bool = True):
        super(ThinNet, self).__init__()

        assert len(filters) >= 1, ''
        assert len(filters) == len(n_blocks), ''
        assert len(filters) == len(n_layers), ''

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.filters = filters
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.bn = bn
        
        self.features = self._make_blocks()
        
        self.avg = nn.AdaptiveAvgPool2d((2, 2))
        self.fc = nn.Linear(self.filters[-1] * 2 * 2, self.n_classes)

    def forward(self, x):
        x = self.features(x)
        
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def _make_blocks(self):
        blocks = []

        for idx, _ in enumerate(self.filters):
            
            blocks.append(ConvBlock(self.in_channels, self.filters[0], 3, bn=self.bn) if idx == 0 else ConvBlock(self.filters[idx - 1], self.filters[idx], 3, bn=self.bn))
            for _ in range(self.n_blocks[idx]):
                blocks.append(DuplicateConvBlock(self.filters[idx], n_layers=self.n_layers[idx], bn=self.bn))
            
        return nn.Sequential(*blocks)