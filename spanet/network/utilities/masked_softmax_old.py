import torch
from torch import Tensor, tensor


@torch.jit.script
def masked_log_sum_exp(x: Tensor,
                       mask: Tensor,
                       dim: int = 1,
                       epsilon: Tensor = tensor(1e-7)) -> Tensor:
    offset = (x * mask).max(dim=dim, keepdim=True).values

    output = torch.exp(x - offset) * mask
    output = output.sum(dim=dim, keepdim=True)
    output = torch.log(output + epsilon)

    return output + offset


@torch.jit.script
def masked_log_softmax(x: Tensor,
                       mask: Tensor,
                       dim: int = 1,
                       epsilon: Tensor = tensor(1e-7)) -> Tensor:
    normalization_term = masked_log_sum_exp(x, mask, dim, epsilon)
    log_mask = torch.log(mask.float())
    return x - normalization_term + log_mask


@torch.jit.script
def training_masked_log_softmax(x: Tensor,
                                mask: Tensor,
                                dim: int = 1,
                                epsilon: Tensor = tensor(1e-6)) -> Tensor:
    normalization_term = masked_log_sum_exp(x, mask, dim, epsilon)
    return (x - normalization_term) * mask


@torch.jit.script
def masked_softmax(x: Tensor,
                   mask: Tensor,
                   dim: int = 1,
                   eps: Tensor = tensor(1e-6, dtype=torch.float)) -> Tensor:
    offset = x.max(dim, keepdim=True).values
    output = torch.exp(x - offset) * mask

    normalizing_sum = output.sum(dim, keepdim=True) + eps
    return output / normalizing_sum