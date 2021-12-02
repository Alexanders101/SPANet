import torch
from torch.nn import functional as F


def masked_log_softmax(
        vector: torch.Tensor,
        mask: torch.BoolTensor,
        dim: int = -1
) -> torch.Tensor:
    """
    Another alternative implementation of the masked log-softmax, this time doing a pure
    mask (setting invalid values to -inf) but also preventing any gradient from flowing
    at all to masked values!
    """

    if mask is not None:
        # Create a -inf with the correct device and datatype
        fill_value = torch.log(vector.new_zeros(()))

        # Replace all masked entries in the output with the gradient-less -inf
        vector = torch.masked_fill(vector, ~mask, fill_value)

    return F.log_softmax(vector, dim=dim)


def masked_softmax(
        vector: torch.Tensor,
        mask: torch.BoolTensor,
        dim: int = -1,
        memory_efficient: bool = False,
) -> torch.Tensor:
    return torch.exp(masked_log_softmax(vector, mask, dim))
