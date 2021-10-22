import cv2 as cv
import numpy as np
import torch
from PIL import Image


class Dilate(torch.nn.Module):
    def __init__(self, radius, shape=cv.MORPH_ELLIPSE):
        super().__init__()
        self.radius = radius
        self.shape = shape
        self.kernel = cv.getStructuringElement(
            shape, (2*radius+1, 2*radius+1), (radius, radius))

    def forward(self, image):
        return cv.morphologyEx(image, cv.MORPH_DILATE, self.kernel)

    def __repr__(self):
        return self.__class__.__name__ + f'(radius={self.radius}, type={self.shape})'


class Erode(torch.nn.Module):
    def __init__(self, radius, shape=cv.MORPH_ELLIPSE):
        super().__init__()
        self.radius = radius
        self.shape = shape
        self.kernel = cv.getStructuringElement(
            shape, (2*radius+1, 2*radius+1), (radius, radius))

    def forward(self, image):
        return cv.morphologyEx(image, cv.MORPH_ERODE, self.kernel)

    def __repr__(self):
        return self.__class__.__name__ + f'(radius={self.radius}, type={self.shape})'


class Open(torch.nn.Module):
    def __init__(self, radius, shape=cv.MORPH_ELLIPSE):
        super().__init__()
        self.radius = radius
        self.shape = shape
        self.kernel = cv.getStructuringElement(
            shape, (2*radius+1, 2*radius+1), (radius, radius))

    def forward(self, image):
        return cv.morphologyEx(image, cv.MORPH_OPEN, self.kernel)

    def __repr__(self):
        return self.__class__.__name__ + f'(radius={self.radius}, type={self.shape})'


class Close(torch.nn.Module):
    def __init__(self, radius, shape=cv.MORPH_ELLIPSE):
        super().__init__()
        self.radius = radius
        self.shape = shape
        self.kernel = cv.getStructuringElement(
            shape, (2*radius+1, 2*radius+1), (radius, radius))

    def forward(self, image):
        return cv.morphologyEx(image, cv.MORPH_CLOSE, self.kernel)

    def __repr__(self):
        return self.__class__.__name__ + f'(radius={self.radius}, type={self.shape})'


class Gradient(torch.nn.Module):
    def __init__(self, radius, shape=cv.MORPH_ELLIPSE):
        super().__init__()
        self.radius = radius
        self.shape = shape
        self.kernel = cv.getStructuringElement(
            shape, (2*radius+1, 2*radius+1), (radius, radius))

    def forward(self, image):
        return cv.morphologyEx(image, cv.MORPH_GRADIENT, self.kernel)

    def __repr__(self):
        return self.__class__.__name__ + f'(radius={self.radius}, type={self.shape})'


class Median(torch.nn.Module):
    def __init__(self, radius):
        super().__init__()
        self.radius = radius
        self.ksize = 2*radius+1

    def forward(self, image):
        return cv.medianBlur(image, 5)

    def __repr__(self):
        return self.__class__.__name__ + f'(ksize={self.ksize})'


class Gaussian(torch.nn.Module):
    def __init__(self, radius, sigma=1.0):
        super().__init__()
        self.radius = radius
        self.sigma = sigma
        self.ksize = 2*radius+1

    def forward(self, image):
        return cv.GaussianBlur(image, [self.ksize, self.ksize], sigmaX=self.sigma)

    def __repr__(self):
        return self.__class__.__name__ + f'(ksize={self.ksize}, sigma={self.sigma})'


class Bilateral(torch.nn.Module):
    def __init__(self, radius, sigma_color, sigma_space):
        super().__init__()
        self.radius = radius
        self.d = 2*radius+1
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def forward(self, image):
        return cv.bilateralFilter(image, self.d, self.sigma_color * 2, self.sigma_space / 2)

    def __repr__(self):
        return self.__class__.__name__ + f'(d={self.d}, sigma_color={self.sigma_color}, sigma_space={self.sigma_space})'


class LocalDestroyer:
    def __init__(self, radius=1) -> None:
        self.radius = radius

        self.ops = [
            # Dilate(self.radius),
            # Erode(self.radius),
            # Open(self.radius),
            # Close(self.radius),
            # Gradient(self.radius),

            # Median(self.radius),
            # Gaussian(self.radius),
            Bilateral(9, 19, 19),
        ]

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, np.ndarray):
            ...
        else:
            raise ValueError(f'Not support type: {type(image)}')

        func = np.random.choice(self.ops, 1)
        image = func[0](image)
    
        return Image.fromarray(image)