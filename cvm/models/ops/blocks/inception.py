import torch
import torch.nn as nn
from typing import List, OrderedDict
from .vanilla_conv2d import Conv2d1x1Block, Conv2dBlock
from .channel import Combine, ConcatBranches


class InceptionA(ConcatBranches):
    def __init__(
        self,
        inp: int,
        planes_1x1: int,
        planes_3x3: List[int],
        planes_3x3db: List[int],
        planes_pool: int
    ) -> None:
        super().__init__(OrderedDict([
            ('branch-1x1', Conv2d1x1Block(inp, planes_1x1)),
            ('branch-3x3', nn.Sequential(
                Conv2d1x1Block(inp, planes_3x3[0]),
                Conv2dBlock(planes_3x3[0], planes_3x3[1], kernel_size=3)
            )),
            ('branch-3x3db', nn.Sequential(
                Conv2d1x1Block(inp, planes_3x3db[0]),
                Conv2dBlock(planes_3x3db[0], planes_3x3db[1], kernel_size=3),
                Conv2dBlock(planes_3x3db[1], planes_3x3db[1], kernel_size=3)
            )),
            ('branch-pool', nn.Sequential(
                nn.AvgPool2d(3, stride=1, padding=1),
                Conv2d1x1Block(inp, planes_pool)
            )),
        ]))


class InceptionB(ConcatBranches):
    def __init__(
        self,
        inp,
        planes_1x1: int,
        planes_7x7: List[int],
        planes_7x7db: List[int],
        planes_pool: int
    ) -> None:
        super().__init__(OrderedDict([
            ('branch-1x1', Conv2d1x1Block(inp, planes_1x1)),
            ('branch-7x7', nn.Sequential(
                Conv2d1x1Block(inp, planes_7x7[0]),
                Conv2dBlock(planes_7x7[0], planes_7x7[1], kernel_size=(1, 7), padding=(0, 3)),
                Conv2dBlock(planes_7x7[1], planes_7x7[2], kernel_size=(7, 1), padding=(3, 0)),
            )),
            ('branch-7x7db', nn.Sequential(
                Conv2d1x1Block(inp, planes_7x7db[0]),
                Conv2dBlock(planes_7x7db[0], planes_7x7db[0], kernel_size=(1, 7), padding=(0, 3)),
                Conv2dBlock(planes_7x7db[0], planes_7x7db[1], kernel_size=(7, 1), padding=(3, 0)),
                Conv2dBlock(planes_7x7db[1], planes_7x7db[1], kernel_size=(1, 7), padding=(0, 3)),
                Conv2dBlock(planes_7x7db[1], planes_7x7db[2], kernel_size=(7, 1), padding=(3, 0)),
            )),
            ('branch-pool', nn.Sequential(
                nn.AvgPool2d(3, stride=1, padding=1),
                Conv2d1x1Block(inp, planes_pool)
            )),
        ]))


class InceptionC(ConcatBranches):
    def __init__(
        self,
        inp,
        planes_1x1: int,
        planes_3x3: List[int],
        planes_3x3db: List[int],
        planes_pool
    ) -> None:
        super().__init__(OrderedDict([
            ('branch-1x1', Conv2d1x1Block(inp, planes_1x1)),
            ('branch-3x3', nn.Sequential(
                Conv2d1x1Block(inp, planes_3x3[0]),
                ConcatBranches(OrderedDict([
                    ('branch-3x3-1', Conv2dBlock(planes_3x3[0], planes_3x3[1], kernel_size=(1, 3), padding=(0, 1))),
                    ('branch-3x3-2', Conv2dBlock(planes_3x3[0], planes_3x3[1], kernel_size=(3, 1), padding=(1, 0)))
                ]))
            )),
            ('branch-3x3db', nn.Sequential(
                Conv2d1x1Block(inp, planes_3x3db[0]),
                Conv2dBlock(planes_3x3db[0], planes_3x3db[1], kernel_size=(1, 3), padding=(0, 1)),
                Conv2dBlock(planes_3x3db[1], planes_3x3db[2], kernel_size=(3, 1), padding=(1, 0)),
                ConcatBranches(OrderedDict([
                    ('branch-3x3db-1', Conv2dBlock(planes_3x3db[2],
                     planes_3x3db[3], kernel_size=(1, 3), padding=(0, 1))),
                    ('branch-3x3db-2', Conv2dBlock(planes_3x3db[2],
                     planes_3x3db[3], kernel_size=(3, 1), padding=(1, 0)))
                ]))
            )),
            ('branch-pool', nn.Sequential(
                nn.AvgPool2d(3, stride=1, padding=1),
                Conv2d1x1Block(inp, planes_pool)
            ))
        ]))


class ReductionA(ConcatBranches):
    def __init__(
        self,
        inp,
        planes_3x3: int,
        planes_3x3db: List[int]
    ):
        super().__init__(OrderedDict([
            ('branch-3x3', Conv2dBlock(inp, planes_3x3, kernel_size=3, stride=2, padding=0)),
            ('branch-3x3db', nn.Sequential(
                Conv2d1x1Block(inp, planes_3x3db[0]),
                Conv2dBlock(planes_3x3db[0], planes_3x3db[1]),
                Conv2dBlock(planes_3x3db[1], planes_3x3db[2], stride=2, padding=0)
            )),
            ('branch-pool', nn.MaxPool2d(3, stride=2))
        ]))


class ReductionB(ConcatBranches):
    def __init__(
        self,
        inp,
        planes_3x3: List[int],
        planes_7x7x3: List[int]
    ) -> None:
        super().__init__(OrderedDict([
            ('branch-3x3', nn.Sequential(
                Conv2d1x1Block(inp, planes_3x3[0]),
                Conv2dBlock(planes_3x3[0], planes_3x3[1], kernel_size=3, stride=2, padding=0),
            )),
            ('branch-7x7x3', nn.Sequential(
                Conv2d1x1Block(inp, planes_7x7x3[0]),
                Conv2dBlock(planes_7x7x3[0], planes_7x7x3[0], kernel_size=(1, 7), padding=(0, 3)),
                Conv2dBlock(planes_7x7x3[0], planes_7x7x3[1], kernel_size=(7, 1), padding=(3, 0)),
                Conv2dBlock(planes_7x7x3[1], planes_7x7x3[1], kernel_size=3, stride=2, padding=0)
            )),
            ('branch-pool', nn.MaxPool2d(3, stride=2))
        ]))


class InceptionResNetA(nn.Module):
    def __init__(
        self,
        inp,
        planes_1x1: int,
        planes_3x3: List[int],
        planes_3x3db: List[int]
    ) -> None:
        super().__init__()

        self.residual = nn.Sequential(
            ConcatBranches(OrderedDict([
                ('branch-1x1', Conv2d1x1Block(inp, planes_1x1)),
                ('branch-3x3', nn.Sequential(
                    Conv2d1x1Block(inp, planes_3x3[0]),
                    Conv2dBlock(planes_3x3[0], planes_3x3[1])
                )),
                ('branch-3x3db', nn.Sequential(
                    Conv2d1x1Block(inp, planes_3x3db[0]),
                    Conv2dBlock(planes_3x3db[0], planes_3x3db[1]),
                    Conv2dBlock(planes_3x3db[1], planes_3x3db[2])
                ))
            ])),
            Conv2d1x1Block(planes_1x1 + planes_3x3[-1] + planes_3x3db[-1], inp)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.residual(x))


class InceptionResNetB(nn.Module):
    def __init__(
        self,
        inp,
        planes_1x1: int,
        planes_7x7: List[int]
    ) -> None:
        super().__init__()

        self.residual = nn.Sequential(
            ConcatBranches(OrderedDict([
                ('branch-1x1', Conv2d1x1Block(inp, planes_1x1)),
                ('branch-7x7', nn.Sequential(
                    Conv2d1x1Block(inp, planes_7x7[0]),
                    Conv2dBlock(planes_7x7[0], planes_7x7[1], kernel_size=(1, 7), padding=(0, 3)),
                    Conv2dBlock(planes_7x7[1], planes_7x7[2], kernel_size=(7, 1), padding=(3, 0)),
                ))
            ])),
            Conv2d1x1Block(planes_1x1 + planes_7x7[-1], inp)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.residual(x))


class InceptionResNetC(nn.Module):
    def __init__(
        self,
        inp,
        planes_1x1: int,
        planes_3x3: List[int]
    ) -> None:
        super().__init__()

        self.residual = nn.Sequential(
            ConcatBranches(OrderedDict([
                ('branch-1x1', Conv2d1x1Block(inp, planes_1x1)),
                ('branch-7x7', nn.Sequential(
                    Conv2d1x1Block(inp, planes_3x3[0]),
                    Conv2dBlock(planes_3x3[0], planes_3x3[1], kernel_size=(1, 3), padding=(0, 1)),
                    Conv2dBlock(planes_3x3[1], planes_3x3[2], kernel_size=(3, 1), padding=(1, 0)),
                ))
            ])),
            Conv2d1x1Block(planes_1x1 + planes_3x3[-1], inp)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.residual(x))


class ReductionC(ConcatBranches):
    def __init__(
        self,
        inp,
        planes_3x3_1: List[int],
        planes_3x3_2: List[int],
        branch_3x3db: List[int]
    ) -> None:
        super().__init__(OrderedDict([
            ('branch-3x3-1', nn.Sequential(
                Conv2d1x1Block(inp, planes_3x3_1[0]),
                Conv2dBlock(planes_3x3_1[0], planes_3x3_1[1], kernel_size=3, stride=2, padding=0),
            )),
            ('branch-3x3-2', nn.Sequential(
                Conv2d1x1Block(inp, planes_3x3_2[0]),
                Conv2dBlock(planes_3x3_2[0], planes_3x3_2[1], kernel_size=3, stride=2, padding=0),
            )),
            ('branch-3x3db', nn.Sequential(
                Conv2d1x1Block(inp, branch_3x3db[0]),
                Conv2dBlock(branch_3x3db[0], branch_3x3db[1], kernel_size=3),
                Conv2dBlock(branch_3x3db[1], branch_3x3db[2], kernel_size=3, stride=2, padding=0)
            )),
            ('branch-pool', nn.MaxPool2d(3, stride=2))
        ]))
