import torch


class DataIterator:
    def __init__(
        self,
        loader,
        type: str = 'dali'
    ):
        self.loader = loader
        self.type = type
        self._counter = 0
        self.itor = self

    def __iter__(self):
        self.itor = iter(self.loader)
        return self

    def __next__(self):
        batch = next(self.itor)

        if self.type == 'dali':
            input = batch[0]["data"]
            target = batch[0]["label"].squeeze(-1).long()
        else:
            input = batch[0].cuda(non_blocking=True)
            target = batch[1].cuda(non_blocking=True)

        return input, target

    @property
    def sampler(self):
        return self.loader.sampler if self.type == 'torch' else None

    @property
    def dataset(self):
        return self.loader.dataset if self.type == 'torch' else None

    def reset(self):
        self._counter += 1

        if self.type == 'dali':
            self.loader.reset()
        elif isinstance(self.sampler, torch.utils.data.distributed.DistributedSampler):
            self.loader.sampler.set_epoch(self._counter)

    def __len__(self):
        return len(self.loader)
