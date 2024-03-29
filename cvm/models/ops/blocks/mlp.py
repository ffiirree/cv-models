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

        super().__init__(
            nn.Linear(in_features, hidden_features),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(dropout_rate)
        )
