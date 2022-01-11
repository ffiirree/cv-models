import torch

__all__ = ['channel_shuffle', 'make_divisible', 'get_gaussian_kernel1d', 'get_gaussian_kernel2d']


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


def get_gaussian_kernel1d(kernel_size, sigma: torch.Tensor):
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size).to(sigma.device)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    return pdf / pdf.sum()


def get_gaussian_kernel2d(kernel_size, sigma: torch.Tensor):
    kernel1d = get_gaussian_kernel1d(kernel_size, sigma)
    return torch.mm(kernel1d[:, None], kernel1d[None, :])
