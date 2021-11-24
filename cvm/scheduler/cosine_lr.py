import math
import warnings
import torch.optim as optim


__all__ = ['WarmUpCosineLR']


class WarmUpCosineLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, steps, min_lr=0.1, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        self.steps = steps - self.warmup_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch < self.warmup_steps:
            return [base_lr * (float(1 + self.last_epoch) / self.warmup_steps) for base_lr in self.base_lrs]

        return [self.min_lr + (base_lr - self.min_lr) * (1 + math.cos(math.pi * (1 + self.last_epoch - self.warmup_steps) / self.steps)) / 2
                for base_lr in self.base_lrs]

    def __repr__(self) -> str:
        return f'WarmUpCosineLR(warmup_steps={self.warmup_steps}, steps={self.steps}, min_lr={self.min_lr})'
