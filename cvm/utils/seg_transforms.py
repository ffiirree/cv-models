import torch
import numpy as np
import torchvision.transforms as T
from torchvision.transforms import functional as TF


class Compose(T.Compose):
    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, images, targets):
        for t in self.transforms:
            images, targets = t(images, targets)
        return images, targets


class PILToTensor:
    def __call__(self, images, targets):
        return TF.to_tensor(images), torch.as_tensor(np.array(targets), dtype=torch.int64)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, images, targets):
        if torch.rand(1) < self.p:
            return TF.hflip(images), TF.hflip(targets)
        return images, targets

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, images, targets):
        if torch.rand(1) < self.p:
            return TF.vflip(images), TF.vflip(targets)
        return images, targets

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ConvertImageDtype(T.ConvertImageDtype):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__(dtype=dtype)

    def forward(self, images, targets):
        return super().forward(images), targets


class Normalize(T.Normalize):
    def __init__(self, mean, std, inplace=False):
        super().__init__(mean, std, inplace)

    def forward(self, images, targets):
        return super().forward(images), targets


class Resize(T.Resize):
    def __init__(self, size, interpolation=TF.InterpolationMode.BILINEAR):
        super().__init__(size, interpolation=interpolation)

    def forward(self, images, targets):
        images = TF.resize(images, self.size, self.interpolation)
        targets = TF.resize(targets, self.size, TF.InterpolationMode.NEAREST)

        return images, targets


class RandomCrop(T.RandomCrop):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__(
            size,
            padding=padding,
            pad_if_needed=pad_if_needed,
            fill=fill,
            padding_mode=padding_mode
        )

    def forward(self, images, targets):
        if self.padding is not None:
            img = TF.pad(images, self.padding, self.fill, self.padding_mode)

        width, height = TF.get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = TF.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = TF.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return TF.crop(img, i, j, h, w), TF.crop(targets, i, j, h, w)


class RandomResizedCrop(T.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=TF.InterpolationMode.BILINEAR):
        super().__init__(size, scale=scale, ratio=ratio, interpolation=interpolation)

    def forward(self, images, targets):
        i, j, h, w = self.get_params(images, self.scale, self.ratio)
        images = TF.resized_crop(images, i, j, h, w, self.size, self.interpolation)
        targets = TF.resized_crop(targets, i, j, h, w, self.size, TF.InterpolationMode.NEAREST)
        return images, targets
