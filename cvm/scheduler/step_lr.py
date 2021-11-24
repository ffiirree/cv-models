import warnings
import torch.optim as optim


__all__ = ['WarmUpStepLR']


class WarmUpStepLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, step_size, gamma=0.1, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch < self.warmup_steps:
            return [base_lr * (float(1 + self.last_epoch) / self.warmup_steps) for base_lr in self.base_lrs]

        milestone = ((self.last_epoch - self.warmup_steps) // self.step_size)
        return [base_lr * self.gamma ** milestone for base_lr in self.base_lrs]

    def __repr__(self) -> str:
        return f'WarmUpStepLR(warmup_steps={self.warmup_steps}, step_size={self.step_size}, gamma={self.gamma})'
