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
import torch.distributed as dist

try:
    import timm
    has_timm = True
except ImportError:
    has_timm = False

__all__ = [
    'Benchmark', 'env_info', 'manual_seed',
    'named_layers', 'accuracy', 'AverageMeter',
    'module_parameters', 'group_params', 'list_models',
    'list_datasets', 'is_dist_avail_and_initialized', 'get_world_size'
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

    torch.set_printoptions(precision=10)


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


def group_params(model, wd: float, no_bias_bn_decay: bool = False):
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
        for m, n, p in module_parameters(model):
            if isinstance(m, nn.modules.batchnorm._BatchNorm) or n == 'bias':
                no_wd_params.append(p)
            else:
                wd_params.append(p)
        return [
            {'params': wd_params, 'weight_decay': wd},
            {'params': no_wd_params, 'weight_decay': 0.}
        ]


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _filter_models(name_list, prefix='', sort=False):
    models = [prefix + name for name in name_list
              if name.islower() and not name.startswith("__")
              and callable(name_list[name])]
    return models if not sort else sorted(models)


def list_models(lib: str = 'all'):
    assert lib in ['all', 'cvm', 'torch', 'timm'], f'Unknown library {lib}.'

    if lib == 'all':
        cvm_models = _filter_models(torchvision.models.__dict__, sort=True)
        torch_models = _filter_models(models.__dict__, 'torch/', True)

        timm_models = [
            'timm/' + name for name in timm.list_models()
        ] if has_timm else []
        return cvm_models + torch_models + timm_models

    elif lib == 'torch':
        return _filter_models(
            torchvision.models.__dict__,
            prefix='torch/',
            sort=True
        )
    elif lib == 'timm':
        assert has_timm, 'Please install timm first.'
        return ['timm/' + name for name in timm.list_models()]
    else:
        return _filter_models(models.__dict__, sort=True)


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
