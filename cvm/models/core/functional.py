import torch

__all__ = ['channel_shuffle', 'make_divisible']

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

def make_divisible(value, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
        
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    
    # Make sure that round down does not go down by more than 10%.
    if new_value < 0.9 * value:
        new_value += divisor
        
    return new_value