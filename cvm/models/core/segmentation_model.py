import torch.nn as nn
from torch.nn import functional as F
from typing import List, Optional

__all__ = ['SegmentationModel']


class SegmentationModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        out_stages: List[int],
        decode_head: nn.Module = None,
        aux_head: Optional[nn.Module] = None
    ):
        super().__init__()

        if out_stages is None:
            out_stages = [4]

        assert aux_head is None or len(out_stages) == 2, ''

        self.backbone = backbone
        self.out_stages = out_stages
        self.decode_head = decode_head
        self.aux_head = aux_head

    def forward(self, x):
        input_size = x.shape[-2:]

        stages = [self.backbone.stem(x)]
        stages.append(self.backbone.stage1(stages[0]))
        stages.append(self.backbone.stage2(stages[1]))
        stages.append(self.backbone.stage3(stages[2]))
        stages.append(self.backbone.stage4(stages[3]))

        decode_out = self.decode_head(stages[self.out_stages[-1]])
        decode_out = F.interpolate(decode_out, size=input_size, mode='bilinear', align_corners=False)

        if self.aux_head:
            aux_out = self.aux_head(stages[self.out_stages[-2]])
            aux_out = F.interpolate(aux_out, size=input_size, mode='bilinear', align_corners=False)
            return (decode_out, aux_out)

        return (decode_out,)
