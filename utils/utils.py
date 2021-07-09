import time
from json import dumps
import torch
import torchvision
import platform
import numpy as np
import random

__all__ = ['Benchmark', 'env_info', 'manual_seed']


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
