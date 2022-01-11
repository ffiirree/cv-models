from functools import partial
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Optional
from cvm.models.core import blocks

from torchvision.models.feature_extraction import create_feature_extractor

__all__ = ['SegmentationModel']


class SegmentationModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        out_stages: List[int],
        decode_head: nn.Module = None,
        aux_head: Optional[nn.Module] = None,
        cls_head: Optional[nn.Module] = None
    ):
        super().__init__()

        if out_stages is None:
            out_stages = [4]

        self.backbone = create_feature_extractor(
            backbone,
            return_nodes=[f'stage{i}' for i in out_stages],
            tracer_kwargs={'leaf_modules': [blocks.Stage]}
        )
        self.out_stages = out_stages
        self.decode_head = decode_head
        self.aux_head = aux_head
        self.cls_head = cls_head
        self.interpolate = partial(F.interpolate, mode='bilinear', align_corners=False)

    def forward(self, x):
        size = x.shape[-2:]

        stages = self.backbone(x)

        out = self.decode_head(stages[f'stage{self.out_stages[-1]}'])
        out = self.interpolate(out, size=size)

        res = {'out': out}

        if self.aux_head:
            aux = self.aux_head(stages[f'stage{self.out_stages[-2]}'])
            aux = self.interpolate(aux, size=size)
            res['aux'] = aux

        if self.cls_head:
            cls = self.cls_head(stages[f'stage{self.out_stages[-1]}'])
            cls = cls.reshape(cls.shape[0], cls.shape[1], 1, 1)
            res['out'] = out * torch.sigmoid(cls)

        return res
