from os import path
import inspect
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision import datasets
import torchvision.transforms as T

from PIL import Image

import cvm
from cvm.data.samplers import RASampler
from cvm.utils.augment import *
from cvm.utils.coco import get_coco
from . import seg_transforms as ST
from .utils import group_params, list_datasets, get_world_size
from cvm.data.constants import *
from cvm.data.loader import DataIterator
from cvm.models.ops import blocks
from functools import partial

from torch.nn.parallel import DistributedDataParallel as DDP

import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from typing import List, Union

try:
    import timm
    has_timm = True
except ImportError:
    has_timm = False

__all__ = [
    'create_model', 'create_optimizer', 'create_scheduler',
    'create_transforms', 'create_dataset', 'create_loader',
    'get_dataset_mean', 'get_dataset_std'
]


@pipeline_def
def create_dali_pipeline(
    data_dir,
    crop_size,
    resize_size,
    shard_id,
    num_shards,
    random_scale=[0.08, 1.0],
    interpolation=types.INTERP_TRIANGULAR,
    antialias=True,
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
            random_area=random_scale,
            num_attempts=100
        )

        images = fn.resize(
            images,
            device=dali_device,
            resize_x=crop_size,
            resize_y=crop_size,
            interp_type=interpolation,
            antialias=antialias
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
            interp_type=interpolation,
            antialias=antialias
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
    cuda: bool = True,
    sync_bn: bool = False,
    distributed: bool = False,
    local_rank: int = 0,
    weights: str='DEFAULT',
    **kwargs
):
    if name.startswith('torch/'):
        name = name.replace('torch/', '')

        _models = torchvision.models
        if len(name.split('/')) > 1:
            _models = torchvision.models.__dict__[name.split('/')[0]]
            name = name.split('/')[1]

        if pretrained:
            model = _models.__dict__[name](weights=weights)
        else:
            model = _models.__dict__[name]()
    elif name.startswith('timm/'):
        assert has_timm, 'Please install timm first.'
        name = name.replace('timm/', '')
        model = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=kwargs.get('num_classes', 1000),
            drop_rate=kwargs.get('dropout_rate', 0.0),
            drop_path_rate=kwargs.get('drop_path_rate', None),
            drop_block_rate=kwargs.get('drop_block', None),
            bn_momentum=kwargs.get('bn_momentum', None),
            bn_eps=kwargs.get('bn_eps', None),
            scriptable=kwargs.get('scriptable', False),
            checkpoint_path=kwargs.get('initial_checkpoint', None),
        )
    else:
        _models = cvm.models
        if len(name.split('/')) > 1:
            _models = cvm.models.__dict__[name.split('/')[0]]
            name = name.split('/')[1]

        if 'bn_eps' in kwargs and kwargs['bn_eps'] and 'bn_momentum' in kwargs and kwargs['bn_momentum']:
            with blocks.normalizer(partial(nn.BatchNorm2d, eps=kwargs['bn_eps'], momentum=kwargs['bn_momentum'])):
                model = _models.__dict__[name](pretrained=pretrained, **kwargs)
        else:
            model = _models.__dict__[name](pretrained=pretrained, **kwargs)

    if cuda:
        model.cuda()

    if sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if distributed:
        model = DDP(model, device_ids=[local_rank])
    return model


def create_optimizer(
    name: str = 'sgd',
    params: nn.Module = None,
    lr: float = 0.1,
    weight_decay: float = 0.0,
    no_bias_bn_wd: bool = False,
    **kwargs
):
    if isinstance(params, nn.Module):
        params = group_params(
            params,
            weight_decay,
            no_bias_bn_wd,
            lr
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
    elif name == 'adamw':
        return optim.AdamW(
            params,
            lr=lr,
            betas=kwargs['adam_betas']
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
    elif name == 'cosine':
        return cvm.scheduler.WarmUpCosineLR(
            optimizer,
            warmup_steps=kwargs['warmup_epochs'] * step_per_epoch,
            steps=kwargs['epochs'] * step_per_epoch,
            min_lr=kwargs['min_lr']
        )

    return None


def _get_name(name):
    if not isinstance(name, str):
        name = name.__class__.__name__
    return name


def _get_dataset_image_size(name):
    name = _get_name(name).lower()

    if name.startswith('cifar'):
        return CIFAR_IMAGE_SIZE
    if 'mnist' in name:
        return MNIST_IMAGE_SIZE
    return 0


def get_dataset_mean(name):
    return _get_dataset_mean_or_std(name, 'mean')


def get_dataset_std(name):
    return _get_dataset_mean_or_std(name, 'std')


def _get_dataset_mean_or_std(name, attr):
    name = _get_name(name).lower()

    if name.startswith('cifar'):
        return CIFAR_MEAN if attr == 'mean' else CIFAR_STD
    if name == 'imagenet':
        return IMAGE_MEAN if attr == 'mean' else IMAGE_STD
    if name == 'mnist':
        return MNIST_MEAN if attr == 'mean' else MNIST_STD
    if name.startswith('voc') or name.startswith('sbd'):
        return VOC_MEAN if attr == 'mean' else VOC_STD
    if name.startswith('coco'):
        return VOC_MEAN if attr == 'mean' else VOC_STD
    return IMAGE_MEAN if attr == 'mean' else IMAGE_STD


_pil_interpolation_to_str = {
    Image.NEAREST: 'nearest',
    Image.BILINEAR: 'bilinear',
    Image.BICUBIC: 'bicubic',
    Image.BOX: 'box',
    Image.HAMMING: 'hamming',
    Image.LANCZOS: 'lanczos',
}
_str_to_pil_interpolation = {b: a for a, b in _pil_interpolation_to_str.items()}


def str_to_pil_interp(mode_str):
    return _str_to_pil_interpolation[mode_str]


def _to_dali_interpolation(interpolation):
    interp_types = {
        'nearest': types.INTERP_NN,
        'linear': types.INTERP_LINEAR,          # (2d data) = bilinear
        'cubic': types.INTERP_CUBIC,
        'triangular': types.INTERP_LINEAR,
        'gaussian': types.INTERP_GAUSSIAN,
        'lanczos': types.INTERP_LANCZOS3,

        # For pytorch compatibility
        'bilinear': types.INTERP_LINEAR,
    }

    return interp_types[interpolation]


def create_transforms(
    resize_size: int = 256,
    crop_size: int = 224,
    random_scale: List[float] = [0.08, 1.0],
    padding: int = 0,
    interpolation=T.InterpolationMode.BILINEAR,
    random_crop: bool = False,
    is_training: bool = True,
    mean=IMAGE_MEAN,
    std=IMAGE_STD,
    hflip: float = 0.5,
    vflip: float = 0.0,
    color_jitter=None,
    augment: str = None,
    random_erasing: float = 0.,
    dataset_image_size: int = 0
):
    ops = []
    if not is_training:
        if dataset_image_size != resize_size:
            ops.append(T.Resize(resize_size, interpolation=interpolation))
        if dataset_image_size != crop_size:
            ops.append(T.CenterCrop(crop_size))
    else:
        # cifar10/100
        if random_crop:
            if dataset_image_size < crop_size:
                ops.append(T.Resize(crop_size, interpolation=interpolation))
            if padding != 0 or dataset_image_size > crop_size:
                ops.append(T.RandomCrop(crop_size, padding))
        # imagenet
        else:
            ops.append(T.RandomResizedCrop(
                crop_size,
                scale=random_scale,
                ratio=(3. / 4., 4. / 3.),
                interpolation=interpolation
            ))

        if hflip > 0.0:
            ops.append(T.RandomHorizontalFlip(hflip))
        if vflip > 0.0:
            ops.append(T.RandomVerticalFlip(vflip))

        if isinstance(crop_size, (tuple, list)):
            img_size_min = min(crop_size)
        else:
            img_size_min = crop_size
        aug_hparams = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        aug_hparams['interpolation'] = str_to_pil_interp(interpolation.value)
        if augment:
            if augment.startswith("rand"):
                ops.append(rand_augment_transform(augment, aug_hparams))
            elif augment.startswith("augmix"):
                ops.append(augment_and_mix_transform(augment, aug_hparams))
            elif augment.startswith('torch/autoaug'):
                if augment == 'torch/autoaug-cifar10':
                    ops.append(T.AutoAugment(T.AutoAugmentPolicy.CIFAR10))
                elif augment == 'torch/autoaug-imagenet':
                    ops.append(T.AutoAugment(T.AutoAugmentPolicy.IMAGENET))
                elif augment == 'torch/autoaug-svhn':
                    ops.append(T.AutoAugment(T.AutoAugmentPolicy.SVHN))
            else:
                ops.append(auto_augment_transform(augment, aug_hparams))

        if color_jitter > 0.0:
            ops.append(T.ColorJitter(*((color_jitter, ) * 3)))  # no hue

    ops.append(T.PILToTensor())
    ops.append(T.ConvertImageDtype(torch.float))
    ops.append(T.Normalize(mean, std))

    if is_training and random_erasing > 0.0:
        ops.append(T.RandomErasing(random_erasing, inplace=True))

    return T.Compose(ops)


def create_segmentation_transforms(
    resize_size: int,
    crop_size: int,
    interpolation=T.InterpolationMode.BILINEAR,
    padding: int = 0,
    is_training: bool = True,
    mean=VOC_MEAN,
    std=VOC_STD,
    hflip: float = 0.5
):
    ops = []
    if is_training:
        # ops.append(ST.RandomCrop(crop_size, pad_if_needed=True, padding=padding))
        ops.append(ST.RandomResizedCrop(crop_size, (0.5, 2.0), interpolation=interpolation))
        if hflip > 0.0:
            ops.append(ST.RandomHorizontalFlip(hflip))
    else:
        ops.append(ST.Resize(resize_size, interpolation=interpolation))

    ops.append(ST.PILToTensor())
    ops.append(ST.ConvertImageDtype(torch.float))
    ops.append(ST.Normalize(mean, std))

    return ST.Compose(ops)


def create_dataset(
    name: str,
    root: str = '',
    is_training: bool = True,
    download: bool = False,
    **kwargs
):
    if name in list_datasets():
        dataset = datasets.__dict__[name]
        params = inspect.signature(dataset.__init__).parameters.keys()

        if 'mode' in params and 'image_set' in params:
            return datasets.__dict__[name](
                path.expanduser(root),
                mode='segmentation',
                image_set='train' if is_training else 'val',
                download=(download and is_training)
            )

        if 'image_set' in params:
            return datasets.__dict__[name](
                path.expanduser(root),
                image_set='train' if is_training else 'val',
                download=(download and is_training)
            )

        return datasets.__dict__[name](
            path.expanduser(root),
            train=is_training,
            download=(download and is_training)
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
    interpolation: str = 'bilinear',
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
    local_rank: int = 0,
    root: str = None,
    distributed: bool = False,
    ra_repetitions: int = 0,
    transform: T.Compose = None,
    taskname: str = 'classification',
    collate_fn=None,
    **kwargs
):
    assert taskname in ['classification', 'segmentation'], f'Unknown task: {taskname}.'
    # Nvidia/DALI
    if dali:
        assert _get_name(dataset).lower() == 'imagenet', ''
        assert ra_repetitions == 0, 'Do not support RepeatedAugmentation when using nvidia/dali.'

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
            random_scale=kwargs.get('random_scale', [0.08, 1.0]),
            interpolation=_to_dali_interpolation(interpolation),
            dali_cpu=dali_cpu,
            shard_id=local_rank,
            num_shards=get_world_size(),
            is_training=is_training,
            hflip=hflip,
            color_jitter=color_jitter,
            random_erasing=random_erasing
        )
        pipe.build()
        return DataIterator(DALIClassificationIterator(
            pipe,
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL
        ), 'dali')
    # Pytorch/Vision
    else:
        if taskname == 'classification':
            transform = transform or create_transforms(
                is_training=is_training,
                random_scale=kwargs.get('random_scale', [0.08, 1.0]),
                interpolation=T.InterpolationMode(interpolation),
                hflip=hflip,
                vflip=vflip,
                color_jitter=color_jitter,
                random_erasing=random_erasing,
                augment=augment,
                mean=kwargs.get('mean', _get_dataset_mean_or_std(dataset, 'mean')),
                std=kwargs.get('std', _get_dataset_mean_or_std(dataset, 'std')),
                random_crop=crop_size <= 128,
                resize_size=val_resize_size,
                crop_size=crop_size if is_training else val_crop_size,
                padding=crop_padding,
                dataset_image_size=_get_dataset_image_size(dataset),
            )
        elif taskname == 'segmentation':
            transform = transform or create_segmentation_transforms(
                is_training=is_training,
                interpolation=T.InterpolationMode(interpolation),
                hflip=hflip,
                mean=kwargs.get('mean', _get_dataset_mean_or_std(dataset, 'mean')),
                std=kwargs.get('std', _get_dataset_mean_or_std(dataset, 'std')),
                resize_size=val_resize_size,
                crop_size=crop_size if is_training else val_crop_size,
            )

        if dataset == 'CocoDetection':
            dataset = get_coco(
                root=root,
                image_set='train' if is_training else 'val',
                transforms=transform
            )
        elif isinstance(dataset, str):
            dataset = create_dataset(
                dataset,
                root=root,
                is_training=is_training,
                **kwargs
            )
            if taskname == 'classification':
                dataset.transform = transform
            elif taskname == 'segmentation':
                dataset.transforms = transform

        sampler = None
        if distributed:
            if ra_repetitions > 0 and is_training:
                sampler = RASampler(dataset, shuffle=True, repetitions=ra_repetitions)
            else:
                sampler = DistributedSampler(dataset, shuffle=is_training)

        return DataIterator(DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=pin_memory,
            sampler=sampler,
            shuffle=((not distributed) and is_training),
            collate_fn=collate_fn,
            drop_last=(is_training and taskname == 'segmentation')
        ), 'torch')
