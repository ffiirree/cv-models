import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision import datasets
import torchvision.transforms as T
import cvm
from .utils import group_params, list_datasets
from cvm.dataset.constants import *
from functools import partial

__all__ = [
    'create_model', 'create_optimizer', 'create_scheduler',
    'create_transforms', 'create_dataset', 'create_loader'
]


def create_model(
    name: str,
    pretrained: bool = False,
    torch: bool = False,
    **kwargs
):
    if torch:
        return torchvision.models.__dict__[name](pretrained=pretrained)

    if 'bn_eps' in kwargs and kwargs['bn_eps'] and 'bn_momentum' in kwargs and kwargs['bn_momentum']:
        with cvm.models.core.blocks.normalizer(partial(nn.BatchNorm2d, eps=kwargs['bn_eps'], momentum=kwargs['bn_momentum'])):
            return cvm.models.__dict__[name](pretrained=pretrained, **kwargs)
    return cvm.models.__dict__[name](pretrained=pretrained, **kwargs)


def create_optimizer(name: str = 'sgd', params: nn.Module = None, lr: float = 0.1,  **kwargs):
    params = group_params(
        params,
        kwargs['weight_decay'],
        kwargs['no_bias_bn_wd']
    )

    if name == 'sgd':
        return optim.SGD(
            params,
            lr=lr,
            momentum=kwargs['momentum'],
            nesterov=kwargs['nesterov']
        )
    elif name == 'rmsprop':
        return optim.RMSprop(
            params,
            lr=lr,
            alpha=kwargs['rmsprop_decay'],
            momentum=kwargs['momentum'],
            eps=kwargs['rmsprop_eps']
        )
    elif name == 'adam':
        return optim.Adam(
            params,
            lr=lr,
            betas=kwargs['adam_betas'],
        )
    else:
        raise ValueError(f'Invalid optimizer: {name}.')


def create_scheduler(
    name: str = 'cosine',
    optimizer: torch.optim.Optimizer = None,
    step_per_epoch: int = 0,
    **kwargs
):
    if name == 'step':
        return cvm.scheduler.WarmUpStepLR(
            optimizer,
            warmup_steps=kwargs['warmup_epochs'] * step_per_epoch,
            step_size=kwargs['lr_decay_epochs'] * step_per_epoch,
            gamma=kwargs['lr_decay_rate']
        )
    else:
        return cvm.scheduler.WarmUpCosineLR(
            optimizer,
            warmup_steps=kwargs['warmup_epochs'] * step_per_epoch,
            steps=kwargs['epochs'] * step_per_epoch,
            min_lr=kwargs['min_lr']
        )


def _get_name(name):
    if not isinstance(name, str):
        name = name.__class__.__name__
    return name


def _get_dataset_image_size(name):
    name = _get_name(name)

    if name.startswith('CIFAR'):
        return CIFAR_IMAGE_SIZE
    if name.contains('MNIST'):
        return MNIST_IMAGE_SIZE
    return 0


def _get_dataset_mean_or_std(name, attr):
    name = _get_name(name)

    if name.startswith('CIFAR'):
        return CIFAR_MEAN if attr == 'mean' else CIFAR_STD
    if name.lower() == 'imagenet':
        return IMAGE_MEAN if attr == 'mean' else IMAGE_STD
    return (0.5,)


def _get_autoaugment_policy(name):
    name = _get_name(name)

    if name.lower() == 'imagenet':
        return T.AutoAugmentPolicy.IMAGENET
    if name.lower().startswith('cifar'):
        return T.AutoAugmentPolicy.CIFAR10
    if name.lower().startswith('svhn'):
        return T.AutoAugmentPolicy.SVHN

    ValueError(f'Unknown AutoAugmentPolicy: {name}.')


def create_transforms(
    resize_size=256,
    crop_size=224,
    padding: int = 0,
    random_crop: bool = False,
    train=True,
    mean=IMAGE_MEAN,
    std=IMAGE_STD,
    hflip=0.5,
    vflip=0.0,
    color_jitter=None,
    augment: str = None,
    randaugment_n=2,
    randaugment_m=5,
    autoaugment_policy='imagenet',
    random_erasing=0.,
    dataset_image_size=0
):
    ops = []
    if not train:
        if dataset_image_size != crop_size:
            ops.append(T.Resize(crop_size))
        if resize_size != crop_size:
            ops.append(T.CenterCrop(crop_size))
    else:
        if random_crop:
            ops.append(T.RandomCrop(crop_size, padding))
        else:
            ops.append(T.RandomResizedCrop(
                crop_size,
                scale=(0.08, 1.0),
                ratio=(3. / 4., 4. / 3.),
                interpolation=T.InterpolationMode.BILINEAR
            ))
        if hflip > 0.0:
            ops.append(T.RandomHorizontalFlip(hflip))
        if vflip > 0.0:
            ops.append(T.RandomVerticalFlip(vflip))
        if color_jitter > 0.0:
            ops.append(T.ColorJitter(*((color_jitter, ) * 3)))  # no hue

        if augment == 'randaugment':
            ops.append(T.RandAugment(randaugment_n, randaugment_m))
        elif augment == 'autoaugment':
            ops.append(T.AutoAugment(autoaugment_policy))

        if random_erasing > 0.0:
            ops.append(T.RandomErasing(random_erasing))

    ops.append(T.PILToTensor())
    ops.append(T.ConvertImageDtype(torch.float))
    ops.append(T.Normalize(mean, std))
    return T.Compose(ops)


def create_dataset(
    name: str,
    root: str = '',
    train: bool = True,
    download: bool = False,
    **kwargs
):
    if name in list_datasets():
        return datasets.__dict__[name](
            os.path.expanduser(root),
            train=train,
            download=download
        )
    elif name == 'ImageNet':
        return datasets.ImageFolder(
            os.path.join(os.path.expanduser(root), 'train' if train else 'val')
        )
    else:
        ValueError(f'Unknown dataset: {name}.')


def create_loader(
    dataset,
    train,
    batch_size,
    workers,
    pin_memory=True,
    crop_padding: int = 4,
    val_resize_size=256,
    val_crop_size=224,
    crop_size=224,
    hflip=0.5,
    vflip=0.0,
    color_jitter=0.0,
    augment: str = None,
    randaugment_n=2,
    randaugment_m=5,
    **kwargs
):
    if isinstance(dataset, str):
        dataset = create_dataset(
            dataset,
            train=train,
            **kwargs
        )

    dataset.transform = create_transforms(
        train=train,
        hflip=hflip,
        vflip=vflip,
        color_jitter=color_jitter,
        augment=augment,
        randaugment_n=randaugment_n,
        randaugment_m=randaugment_m,
        autoaugment_policy=_get_autoaugment_policy(dataset),
        mean=_get_dataset_mean_or_std(dataset, 'mean'),
        std=_get_dataset_mean_or_std(dataset, 'std'),
        random_crop=crop_size <= 128,
        resize_size=val_resize_size,
        crop_size=crop_size if train else val_crop_size,
        padding=crop_padding,
        dataset_image_size=_get_dataset_image_size(dataset),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=DistributedSampler(dataset, shuffle=train)
    )
