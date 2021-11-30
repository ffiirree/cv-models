""" Real labels evaluator for ImageNet
    [1] Are we done with ImageNet?. arXiv:2006.07159
"""
import os
import json
import torch
import numpy as np

__all__ = ['ImageNet1KRealLabelsEvaluator']


class ImageNet1KRealLabelsEvaluator:

    def __init__(self, samples, labels_file='real_labels.json', topk=(1, 5)):
        with open(labels_file) as f:
            self.labels = {
                f'ILSVRC2012_val_{i + 1:08d}.JPEG': labels for i, labels in enumerate(json.load(f))
            }

        assert len(samples) == len(self.labels)

        self.samples = samples
        self.topk = topk
        self.res = {k: [] for k in topk}
        self.index = 0

    def put(self, output: torch.Tensor):
        maxk = max(self.topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.cpu().numpy()

        for topk_label in pred:
            filename = os.path.basename(self.samples[self.index][0])

            if self.labels[filename]:
                for k in self.topk:
                    self.res[k].append(
                        any([p in self.labels[filename] for p in topk_label[:k]])
                    )
            self.index += 1

    @property
    def accuracy(self):
        return {k: float(np.mean(self.res[k])) * 100 for k in self.topk}
