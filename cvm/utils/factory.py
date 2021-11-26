from os import path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision import datasets
import torchvision.transforms as T
import cvm
from .utils import group_params, list_datasets, get_world_size
from cvm.dataset.constants import *
from functools import partial

from torch.nn.parallel import DistributedDataParallel as DDP

import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

__all__ = [
    'create_model', 'create_optimizer', 'create_scheduler',
    'create_transforms', 'create_dataset', 'create_loader'
]


@pipeline_def
def create_dali_pipeline(
    data_dir,
    crop_size,
    resize_size,
    shard_id,
    num_shards,
    dali_cpu=False,
    is_training=True,
    hflip=0.5,
    color_jitter=0.0,
    random_erasing=0.0,
):
    images, labels = fn.readers.file(
        file_root=data_dir,
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=is_training,
        pad_last_batch=True,
        name="Reader"
    )

    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0

    if is_training:
        images = fn.decoders.image_random_crop(
            images,
            device=decoder_device,
            output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
            preallocate_width_hint=preallocate_width_hint,
            preallocate_height_hint=preallocate_height_hint,
            random_aspect_ratio=[3/4, 4/3],
            random_area=[0.08, 1.0],
            num_attempts=100
        )

        images = fn.resize(
            images,
            device=dali_device,
            resize_x=crop_size,
            resize_y=crop_size,
            interp_type=types.INTERP_TRIANGULAR
        )

        if color_jitter > 0.0:
            images = fn.color_twist(
                images,
                device=dali_device,
                brightness=fn.random.uniform(
                    range=[1 - color_jitter, 1.0 + color_jitter]),
                contrast=fn.random.uniform(
                    range=[1 - color_jitter, 1.0 + color_jitter]),
                saturation=fn.random.uniform(
                    range=[1 - color_jitter, 1.0 + color_jitter])
            )

        mirror = fn.random.coin_flip(probability=hflip) if hflip > 0. else None

    else:
        images = fn.decoders.image(
            images,
            device=decoder_device,
            output_type=types.RGB
        )

        images = fn.resize(
            images,
            device=dali_device,
            size=resize_size,
            mode="not_smaller",
            interp_type=types.INTERP_TRIANGULAR
        )

        mirror = False

    images = fn.crop_mirror_normalize(
        images.gpu(),
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(crop_size, crop_size),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=mirror
    )

    labels = labels.gpu()
    return images, labels


def create_model(
    name: str,
    pretrained: bool = False,
    torch: bool = False,
    cuda: bool = True,
    sync_bn: bool = False,
    distributed: bool = False,
    local_rank: int = 0,
    **kwargs
):
    if torch:
        model = torchvision.models.__dict__[name](pretrained=pretrained)
    else:
        if 'bn_eps' in kwargs and kwargs['bn_eps'] and 'bn_momentum' in kwargs and kwargs['bn_momentum']:
            with cvm.models.core.blocks.normalizer(partial(nn.BatchNorm2d, eps=kwargs['bn_eps'], momentum=kwargs['bn_momentum'])):
                model = cvm.models.__dict__[name](pretrained=pretrained, **kwargs)
        model = cvm.models.__dict__[name](pretrained=pretrained, **kwargs)

    if cuda:
        model.cuda()

    if sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if distributed:
        model = DDP(model, device_ids=[local_rank])
    return model


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
    if 'MNIST' in name:
        return MNIST_IMAGE_SIZE
    return 0


def _get_dataset_mean_or_std(name, attr):
    name = _get_name(name)

    if name.startswith('CIFAR'):
        return CIFAR_MEAN if attr == 'mean' else CIFAR_STD
    if name.lower() == 'imagenet':
        return IMAGE_MEAN if attr == 'mean' else IMAGE_STD
    return IMAGE_MEAN if attr == 'mean' else IMAGE_STD


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
    is_training=True,
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
    if not is_training:
        if dataset_image_size != resize_size:
            ops.append(T.Resize(resize_size))
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
    is_training: bool = True,
    download: bool = False,
    **kwargs
):
    if name in list_datasets():
        return datasets.__dict__[name](
            path.expanduser(root),
            train=is_training,
            download=download
        )
    elif name == 'ImageNet':
        return datasets.ImageFolder(
            path.join(path.expanduser(root), 'train' if is_training else 'val')
        )
    else:
        ValueError(f'Unknown dataset: {name}.')


def create_loader(
    dataset,
    is_training: bool = True,
    batch_size: int = 256,
    workers: int = 4,
    pin_memory: bool = True,
    crop_padding: int = 4,
    val_resize_size: int = 256,
    val_crop_size: int = 224,
    crop_size: int = 224,
    hflip: float = 0.5,
    vflip: float = 0.0,
    color_jitter: float = 0.0,
    random_erasing: float = 0.0,
    dali: bool = False,
    dali_cpu: bool = True,
    augment: str = None,
    randaugment_n=2,
    randaugment_m=5,
    local_rank: int = 0,
    root: str = None,
    distributed: bool = False,
    **kwargs
):
    # Nvidia/DALI
    if dali:
        assert _get_name(dataset).lower() == 'imagenet', ''

        pipe = create_dali_pipeline(
            batch_size=batch_size,
            num_threads=workers,
            device_id=local_rank,
            seed=12 + local_rank,
            data_dir=path.join(
                path.expanduser(root), 'train' if is_training else 'val'
            ),
            crop_size=crop_size if is_training else val_crop_size,
            resize_size=val_resize_size,
            dali_cpu=dali_cpu,
            shard_id=local_rank,
            num_shards=get_world_size(),
            is_training=is_training,
            hflip=hflip,
            color_jitter=color_jitter,
            random_erasing=random_erasing
        )
        pipe.build()
        return DALIClassificationIterator(
            pipe,
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL
        )
    # Pytorch/Vision
    else:
        if isinstance(dataset, str):
            dataset = create_dataset(
                dataset,
                root=root,
                is_training=is_training,
                **kwargs
            )

        dataset.transform = create_transforms(
            is_training=is_training,
            hflip=hflip,
            vflip=vflip,
            color_jitter=color_jitter,
            random_erasing=random_erasing,
            augment=augment,
            randaugment_n=randaugment_n,
            randaugment_m=randaugment_m,
            autoaugment_policy=_get_autoaugment_policy(dataset),
            mean=_get_dataset_mean_or_std(dataset, 'mean'),
            std=_get_dataset_mean_or_std(dataset, 'std'),
            random_crop=crop_size <= 128,
            resize_size=val_resize_size,
            crop_size=crop_size if is_training else val_crop_size,
            padding=crop_padding,
            dataset_image_size=_get_dataset_image_size(dataset),
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=pin_memory,
            sampler=DistributedSampler(
                dataset, shuffle=is_training
            ) if distributed else None,
            shuffle=((not distributed) and is_training)
        )
