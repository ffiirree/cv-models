import torch

__all__ = ['accuracy', 'accuracy_k', 'ConfusionMatrix']


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


def accuracy_k(output: torch.Tensor, target):

    with torch.inference_mode():
        output = output.max(dim=1)[1]
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        mask = output.eq(target)

        return target[mask]


class ConfusionMatrix:
    def __init__(self, num_classes, eps=1e-6):
        self.n = num_classes
        self.mat = None
        self.eps = eps

    def update(self, pr, gt):
        if self.mat is None:
            self.mat = torch.zeros(
                (self.n, self.n), dtype=torch.int64, device=pr.device)

        with torch.inference_mode():
            k = (gt >= 0) & (gt < self.n)
            inds = self.n * gt[k].to(torch.int64) + pr[k]
            self.mat += torch.bincount(inds, minlength=self.n ** 2).reshape(self.n, self.n)

    def all_reduce(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return

        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    @property
    def intersection(self):
        return torch.diag(self.mat)

    @property
    def union(self):
        return self.mat.sum(0) + self.mat.sum(1)

    @property
    def iou(self):
        return (self.intersection / (self.union - self.intersection + self.eps)).tolist()

    @property
    def mean_iou(self):
        return (self.intersection / (self.union - self.intersection + self.eps)).mean().item()

    @property
    def pa(self):
        return (self.intersection.sum() / self.mat.sum()).item()

    @property
    def mean_pa(self):
        return (self.intersection / self.mat.sum(1)).tolist()
