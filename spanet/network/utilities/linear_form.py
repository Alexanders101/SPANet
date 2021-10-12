from typing import List

import torch
from torch import Tensor


@torch.jit.script
def contract_4d(weights: Tensor, x: Tensor) -> Tensor:
    factor = torch.sqrt(weights.shape[0])
    y = torch.einsum('ijkl,bxi->jklbx', weights, x) / factor
    y = torch.einsum('jklbx,byj->klbxy', y, x) / factor
    y = torch.einsum('klbxy,bzk->lbxyz', y, x) / factor
    y = torch.einsum('lbxyz,bwl->bxyzw', y, x) / factor

    return y


@torch.jit.script
def contract_3d(weights: Tensor, x: Tensor) -> Tensor:
    factor = torch.sqrt(weights.shape[0])
    y = torch.einsum('ijk,bxi->jkbx', weights, x) / factor
    y = torch.einsum('jkbx,byj->kbxy', y, x) / factor
    y = torch.einsum('kbxy,bzk->bxyz', y, x) / factor
    return y


@torch.jit.script
def contract_2d(weights: Tensor, x: Tensor) -> Tensor:
    factor = torch.sqrt(weights.shape[0])
    y = torch.einsum('ij,bxi->jbx', weights, x) / factor
    y = torch.einsum('jbx,byj->bxy', y, x) / factor
    return y


@torch.jit.script
def contract_1d(weights: Tensor, x: Tensor) -> Tensor:
    factor = torch.sqrt(weights.shape[0])
    y = torch.einsum('i,bxi->bx', weights, x) / factor
    return y


@torch.jit.script
def contract_linear_form(weights: Tensor, x: Tensor) -> Tensor:
    if weights.ndim == 4:
        return contract_4d(weights, x)
    elif weights.ndim == 3:
        return contract_3d(weights, x)
    elif weights.ndim == 2:
        return contract_2d(weights, x)
    else:
        return contract_1d(weights, x)


@torch.jit.script
def symmetric_tensor(weights: Tensor, permutation_group: List[List[int]]):
    symmetric_weights: Tensor = weights

    for sigma in permutation_group:
        symmetric_weights = symmetric_weights + weights.permute(sigma)

    return symmetric_weights / torch.scalar_tensor(len(permutation_group) + 1)


# A dynamically created symmetric tensor function.
# This is necessary for onnx export since it currently does not
# support tensor.permute with non-constant arguments.
def create_symmetric_function(permutation_group: List[List[int]]):
    code = [
        "def symmetrize_tensor(weights):",
        "    symmetric_weights = weights"
        "    "
    ]

    for sigma in permutation_group:
        code.append(f"    symmetric_weights = symmetric_weights + weights.permute({','.join(map(str, sigma))})")

    code.append(f"    return symmetric_weights / {len(permutation_group) + 1}")
    code = "\n".join(code)

    environment = globals().copy()
    exec(code, environment)

    return environment["symmetrize_tensor"]


@torch.jit.script
def batch_symmetric_tensor(inputs: Tensor, permutation_group: List[List[int]]):
    symmetric_outputs: Tensor = inputs

    for sigma in permutation_group:
        for i in range(inputs.shape[0]):
            symmetric_outputs[i] = symmetric_outputs[i] + inputs[i].permute(sigma)

    return symmetric_outputs / torch.scalar_tensor(len(permutation_group) + 1)
