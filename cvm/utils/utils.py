import os
import time
import random
import torch
from torch import nn
import torchvision
from torchvision import datasets
import platform
import numpy as np
from json import dumps
from cvm import models
from cvm.models.core import blocks
import torch.distributed as dist

try:
    import timm
    has_timm = True
except ImportError:
    has_timm = False

__all__ = [
    'Benchmark', 'env_info', 'manual_seed',
    'named_layers', 'AverageMeter',
    'module_parameters', 'group_params', 'list_models',
    'list_datasets', 'is_dist_avail_and_initialized', 'get_world_size',
    'init_distributed_mode', 'mask_to_label'
]


class Benchmark:
    def __init__(self) -> None:
        self._start = None
        self.start()

    def start(self):
        self._start = time.time()

    def elapsed(self):
        _now = time.time()
        _elapsed = _now - self._start
        self._start = _now

        return _elapsed


def env_info(json: bool = False):
    kvs = {
        'Python': platform.python_version(),
        'torch': torch.__version__,
        'torchvision': torchvision.__version__,
        'CUDA': torch.version.cuda,
        'cuDNN': torch.backends.cudnn.version(),
        'GPU': {
            f'#{i}': {
                'name': torch.cuda.get_device_name(i),
                'memory': f'{torch.cuda.get_device_properties(i).total_memory / (1024 * 1024 * 1024):.2f}GB'
            }
            for i in range(torch.cuda.device_count())
        },
        'Platform': {
            'system': platform.system(),
            'node': platform.node(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        }
    }

    return kvs if not json else dumps(kvs, indent=4, separators=(',', ':'))


def manual_seed(seed: int = 0):
    r"""
        https://pytorch.org/docs/stable/notes/randomness.html
        https://discuss.pytorch.org/t/random-seed-initialization/7854/20
    """
    # numpy
    np.random.seed(seed)
    # Python
    random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # torch.set_printoptions(precision=10)


def named_layers(module, memo=None, prefix: str = ''):
    if memo is None:
        memo = set()
    if module not in memo:
        memo.add(module)
        if not module._modules.items():
            yield prefix, module
        for name, module in module._modules.items():
            if module is None:
                continue
            submodule_prefix = prefix + ('.' if prefix else '') + name
            for m in named_layers(module, memo, submodule_prefix):
                yield m


def module_parameters(model):
    memo = set()

    for _, module in model.named_modules():
        for k, v in module._parameters.items():
            if v is None or v in memo:
                continue
            memo.add(v)
            yield module, k, v


def group_params(model, wd: float, no_bias_bn_decay: bool = False, lr: float = 0.1):
    '''As pointed out by Jia et al. 
    Highly scalable deep learning training system with mixed-precision: Training imagenet in four minutes, 
    however, it’s recommended to only apply the regularization
    to weights to avoid overfitting. The no bias decay
    heuristic follows this recommendation, it only applies
    the weight decay to the weights in convolution and fullyconnected
    layers. Other parameters, including the biases 
    and $\alpha$ and $\beta$  in BN layers, are left unregularized.
    .'''
    if not no_bias_bn_decay:
        return [{'params': model.parameters(), 'weight_decay': wd}]
    else:
        wd_params = []
        no_wd_params = []
        groups = []
        for m, n, p in module_parameters(model):
            if isinstance(m, (nn.modules.batchnorm._BatchNorm)) or n == 'bias':
                no_wd_params.append(p)
            elif isinstance(m, (blocks.SemanticallyDeterminedDepthwiseConv2d)):
                groups.append({'params': p, 'weight_decay': 0.0, 'lr': lr / (m.in_channels // 8)})
            elif isinstance(m, (blocks.GaussianBlur)):
                groups.append({'params': p, 'weight_decay': 0.0, 'lr': lr / m.channels})
            else:
                wd_params.append(p)
        return [
            {'params': wd_params, 'weight_decay': wd, 'lr': lr},
            {'params': no_wd_params, 'weight_decay': 0., 'lr': lr}
        ] + groups


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self._total = 0
        self._count = 0

    def update(self, value, n=1):
        self._total += value * n
        self._count += n

    @property
    def avg(self):
        reduced = reduce_across_processes([self._count, self._total])
        return reduced.tolist()[1] / reduced.tolist()[0]

    @property
    def total(self):
        return reduce_across_processes(self._total).item()

    @property
    def count(self):
        return reduce_across_processes(self._count).item()


def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t


def _filter_models(module, prefix='', sort=True):
    name_list = module.__dict__
    models = [prefix + name for name in name_list
              if name.islower() and not name.startswith("__")
              and callable(name_list[name])]
    return models if not sort else sorted(models)


def list_models(lib: str = 'all'):
    assert lib in ['all', 'cvm', 'torch', 'timm'], f'Unknown library {lib}.'

    cvm_models = [
        *_filter_models(models),
        *_filter_models(models.seg, 'seg/'),
        *_filter_models(models.gan, 'gan/'),
        *_filter_models(models.vae, 'vae/')
    ]
    if lib == 'cvm':
        return cvm_models

    torch_models = [
        *_filter_models(torchvision.models, 'torch/'),
        *_filter_models(torchvision.models.segmentation, 'torch/segmentation/'),
        *_filter_models(torchvision.models.detection, 'torch/detection/'),
        *_filter_models(torchvision.models.video, 'torch/video/'),
    ]
    if lib == 'torch':
        return torch_models

    timm_models = [
        'timm/' + name for name in timm.list_models()
    ] if has_timm else []

    if lib == 'timm':
        return timm_models

    return cvm_models + torch_models + timm_models


def list_datasets():
    _datasets = sorted(
        name for name in datasets.__dict__
        if callable(datasets.__dict__[name])
    )
    _datasets.remove('ImageNet')
    _datasets.remove('ImageFolder')
    _datasets.remove('DatasetFolder')
    return _datasets


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def init_distributed_mode(args):
    args.rank = 0
    args.local_rank = 0
    args.distributed = False

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])

        args.batch_size = int(args.batch_size / torch.cuda.device_count())

        torch.cuda.set_device(args.local_rank)
        args.dist_backend = "nccl"
        args.dist_url = "env://"

        torch.distributed.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
        args.distributed = True
        return True

    return False


def mask_to_label(masks, num_classes):
    labels = torch.zeros((masks.shape[0], num_classes), dtype=masks.dtype, device=masks.device)
    for i in range(masks.shape[0]):
        for j in range(num_classes):
            labels[i][j] = bool((masks[i] == j).sum())
    return labels.float()
