import inspect
import math
import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image


__all__ = ['MixUp', 'RandAugment']


class MixUp(object):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.lam = 1

    def mix(self, x: torch.Tensor, y: torch.Tensor):
        self.lam = np.random.beta(self.alpha, self.alpha)
        indices = torch.randperm(x.shape[0]).to(x.device)
        x = self.lam * x + (1 - self.lam) * x[indices, :]
        y = self.lam * y + (1 - self.lam) * y[indices, :]
        return x, y

    def __repr__(self) -> str:
        return f'MixUp(alpha={self.alpha})'


def _blend(img1, img2, factor):
    factor = float(factor)

    if factor == 0.0:
        return img1
    if factor == 1.0:
        return img2

    bound = 1.0 if img1.is_floating_point() else 255.0
    return ((1.0 - factor) * img1 + factor * img2).clamp(0, bound).to(img1.dtype)


def _get_image_size(img):
    if isinstance(img, torch.Tensor):
        assert img.ndim >= 2, 'Not an image.'
        return [img.shape[-1], img.shape[-2]]

    if isinstance(img, Image.Image):
        return img.size


def _t_solarize_add(img, addition, threshold=128):
    assert isinstance(img, torch.Tensor), 'img is not a torch tensor.'

    bound = 1.0 if img.is_floating_point() else 255.0
    addition = addition / 255.0 if img.is_floating_point() else addition
    threshold = threshold / 255.0 if img.is_floating_point() else threshold

    added_img = (img + addition).clamp(0, bound).to(img.dtype)
    return torch.where(img < threshold, added_img, img)


def _pil_solarize_add(img, addition, threshold=128):
    lut = []
    for i in range(256):
        if i < threshold:
            lut.append(min(255, i + addition))
        else:
            lut.append(i)

    if img.mode in ("L", "RGB"):
        if img.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return img.point(lut)
    else:
        raise img


def solarize_add(img, addition, threshold=128):
    if isinstance(img, torch.Tensor):
        return _t_solarize_add(img, addition, threshold)
    else:
        return _pil_solarize_add(img, addition, threshold)


def color(img: torch.Tensor, factor: float):
    degenerate = TF.rgb_to_grayscale(img, 3)
    if isinstance(img, Image.Image):
        return Image.blend(degenerate, img, factor)
    else:
        return _blend(degenerate, img, factor)


def shear_x(img: torch.Tensor, level, fill=None):
    if isinstance(img, Image.Image):
        return img.transform(img.size, Image.AFFINE, (1, level, 0, 0, 1, 0))

    return TF.affine(
        img,
        angle=0,
        translate=[-img.shape[1] * level / 2, 0],
        scale=1.0,
        shear=[math.degrees(math.atan(level)), 0],
        fill=fill
    )


def shear_y(img: torch.Tensor, level, fill=None):
    if isinstance(img, Image.Image):
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, level, 1, 0))

    return TF.affine(
        img,
        angle=0,
        translate=[0, -img.shape[2] * level / 2],
        scale=1.0,
        shear=[0, math.degrees(math.atan(level))],
        fill=fill
    )


def translate_x(img: torch.Tensor, level, fill=None):
    return TF.affine(
        img,
        angle=0,
        translate=[level, 0],
        scale=1.0,
        shear=[0.0, 0.0],
        fill=fill
    )


def translate_y(img: torch.Tensor, level, fill=None):
    return TF.affine(
        img,
        angle=0,
        translate=[0, level],
        scale=1.0,
        shear=[0.0, 0.0],
        fill=fill
    )


def _pil_erase(img, l, t, r, b, fill):
    arr = np.array(img)
    arr[t:b, l:r, :] = fill
    return Image.fromarray(arr)


def _t_erase(img, l, t, r, b, fill):
    img[:, t:b, l:r] = fill
    return img


def cutout(img: torch.Tensor, pad_size: int, fill=0):
    h, w = _get_image_size(img)

    cutout_center_h = torch.randint(0, h, [1]).item()
    cutout_center_w = torch.randint(0, w, [1]).item()

    t = max(0, cutout_center_h - pad_size)
    b = min(h, cutout_center_h + pad_size)
    l = max(0, cutout_center_w - pad_size)
    r = min(w, cutout_center_w + pad_size)

    if isinstance(img, torch.Tensor):
        return _t_erase(img, l, t, r, b, fill)
    else:
        return _pil_erase(img, l, t, r, b, fill)


NAME_TO_FUNC = {
    'AutoContrast': TF.autocontrast,
    'Equalize': TF.equalize,    # only torch.uint8
    'Invert': TF.invert,
    'Rotate': TF.rotate,
    'Posterize': TF.posterize,  # only torch.uint8
    'Solarize': TF.solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': TF.adjust_contrast,
    'Brightness': TF.adjust_brightness,
    'Sharpness': TF.adjust_sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x,
    'TranslateY': translate_y,
    'Cutout': cutout,
}

_MAX_LEVEL = 10.


def augment_list():
    return [
        'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize',
        'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness',
        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Cutout', 'SolarizeAdd'
    ]


def _randomly_negate(tensor):
    """With 50% prob turn the tensor negative."""
    return -tensor if torch.rand((1)).item() > 0.5 else tensor


def _enhance_level_to_arg(level):
    return ((level/_MAX_LEVEL) * 1.8 + 0.1,)


def _shear_level_to_arg(level):
    level = (level/_MAX_LEVEL) * 0.3
    # Flip level to negative with 50% chance.
    level = _randomly_negate(level)
    return (level,)


def _translate_level_to_arg(level, translate_const):
    level = (level/_MAX_LEVEL) * float(translate_const)
    # Flip level to negative with 50% chance.
    level = _randomly_negate(level)
    return (level,)


def _rotate_level_to_arg(level):
    level = (level/_MAX_LEVEL) * 30.
    level = _randomly_negate(level)
    return (level,)


def level_to_arg():
    return {
        'AutoContrast': lambda level: (),
        'Equalize': lambda level: (),
        'Invert': lambda level: (),
        'Rotate': _rotate_level_to_arg,
        'Posterize': lambda level: (int((level/_MAX_LEVEL) * 4),),
        'Solarize': lambda level: (int((level/_MAX_LEVEL) * 256),),
        'SolarizeAdd': lambda level: (int((level/_MAX_LEVEL) * 110),),
        'Color': _enhance_level_to_arg,
        'Contrast': _enhance_level_to_arg,
        'Brightness': _enhance_level_to_arg,
        'Sharpness': _enhance_level_to_arg,
        'ShearX': _shear_level_to_arg,
        'ShearY': _shear_level_to_arg,
        'Cutout': lambda level: (int((level/_MAX_LEVEL) * 40),),
        # pylint:disable=g-long-lambda
        'TranslateX': lambda level: _translate_level_to_arg(level, 100),
        'TranslateY': lambda level: _translate_level_to_arg(level, 100),
        # pylint:enable=g-long-lambda
    }


class RandAugment:
    def __init__(self, N, M):
        self.n = N
        self.m = M

        self.augment_list = augment_list()
        self.replace_value = [128] * 3

    def __call__(self, image):
        for _ in range(self.n):
            op_to_select = torch.randint(0, len(self.augment_list), [1]).item()
            random_m = float(self.m)

            for i, op_name in enumerate(self.augment_list):
                prob = torch.zeros(1).uniform_(0.2, 0.8)

                func = NAME_TO_FUNC[op_name]
                args = level_to_arg()[op_name](random_m)
                kwargs = dict()

                if 'prop' in inspect.signature(func).parameters:
                    kwargs['prop'] = prob
                if 'fill' in inspect.signature(func).parameters:
                    kwargs['fill'] = self.replace_value

                if op_to_select == i:
                    image = func(image, *args, **kwargs)

        return image
