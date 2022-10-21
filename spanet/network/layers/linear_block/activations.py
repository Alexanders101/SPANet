import torch
from torch import nn


def create_activation(activation: str, input_dim: int) -> nn.Module:
    activation = activation.lower().replace("_", "").replace(" ", "")

    if activation == "relu":
        return nn.ReLU()
    elif activation == "prelu":
        return nn.PReLU(input_dim)
    elif activation == "elu":
        return nn.ELU()
    elif activation == "celu":
        return nn.CELU()
    elif activation == "gelu":
        return nn.GELU()
    else:
        return nn.Identity()


def create_dropout(dropout: float) -> nn.Module:
    if dropout > 0:
        return nn.Dropout(dropout)
    else:
        return nn.Identity()


class ZeroModule(nn.Module):
    def forward(self, vectors):
        return torch.zeros_like(vectors)


def create_residual_connection(skip_connection: bool, input_dim: int, output_dim: int) -> nn.Module:
    if input_dim == output_dim or not skip_connection:
        return nn.Identity()

    return nn.Linear(input_dim, output_dim)
