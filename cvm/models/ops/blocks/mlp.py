from torch import nn
from typing import OrderedDict


class MlpBlock(nn.Sequential):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation_fn: nn.Module = nn.GELU,
        dropout_rate: float = 0.
    ):
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        layers = OrderedDict([
            ('fc1', nn.Linear(in_features, hidden_features)),
            ('act', activation_fn()),
        ])

        if dropout_rate != 0.:
            layers['do1'] = nn.Dropout(dropout_rate)

        layers['fc2'] = nn.Linear(hidden_features, out_features)

        if dropout_rate != 0.:
            layers['do2'] = nn.Dropout(dropout_rate)

        super().__init__(layers)
