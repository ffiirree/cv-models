import torch
from torch import fft

__all__ = ['channel_shuffle', 'make_divisible',
           'get_gaussian_kernel1d', 'get_gaussian_kernel2d',
           'get_gaussian_bandpass_kernel2d', 'get_gaussian_kernels2d',
           'get_distance_grid', 'spectral_filter']


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


def get_gaussian_kernel1d(kernel_size, sigma: float, normalize: bool = True):
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    return pdf / pdf.sum() if normalize else pdf


def get_gaussian_kernel2d(kernel_size, sigma: float, normalize: bool = True):
    ksize_half = (kernel_size - 1) * 0.5

    xs = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    ys = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    x, y = torch.meshgrid(xs, ys, indexing='xy')

    pdf = torch.exp(-0.5 * ((x * x + y * y) / (sigma * sigma)))

    return pdf / pdf.sum() if normalize else pdf


def get_gaussian_bandpass_kernel2d(kernel_size, sigma: float, W: float):
    ksize_half = (kernel_size - 1) * 0.5

    xs = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    ys = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    x, y = torch.meshgrid(xs, ys, indexing='xy')

    d2 = x * x + y * y
    d = torch.sqrt(d2)

    return torch.exp(-((d2 - sigma * sigma) / (d * W)).pow(2))


def get_gaussian_kernels2d(kernel_size, sigma: torch.Tensor, normalize: bool = True):
    ksize_half = (kernel_size - 1) * 0.5

    xs = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    ys = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    x, y = torch.meshgrid(xs, ys, indexing='xy')

    pdf = torch.exp(-0.5 * ((x * x + y * y).repeat(sigma.shape) / torch.pow(sigma, 2)))

    return pdf / pdf.sum([-2, -1], keepdim=True) if normalize else pdf


def get_distance_grid(size):
    size_half = (size - 1) * 0.5

    xs = torch.linspace(-size_half, size_half, steps=size)
    ys = torch.linspace(-size_half, size_half, steps=size)

    x, y = torch.meshgrid(xs, ys, indexing='xy')

    return torch.sqrt(x * x + y * y)


def spectral_filter(x, callback):
    fre_x = fft.fftshift(fft.fft2(x))

    fre_x = callback(fre_x)

    return fft.ifft2(fft.ifftshift(fre_x)).real
