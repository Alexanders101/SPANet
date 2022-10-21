from typing import List, Tuple

import torch
from torch import nn

from spanet.options import Options
from spanet.network.utilities.group_theory import complete_indices, symmetry_group


# noinspection SpellCheckingInspection
class SymmetricAttentionBase(nn.Module):
    WEIGHTS_INDEX_NAMES = "ijklmn"
    INPUT_INDEX_NAMES = "xyzwuv"
    DEFAULT_JET_COUNT = 16

    def __init__(self,
                 options: Options,
                 degree: int,
                 permutation_indices: List[Tuple[int, ...]] = None,
                 attention_dim: int = None) -> None:
        super(SymmetricAttentionBase, self).__init__()

        self.attention_dim = attention_dim
        if attention_dim is None:
            self.attention_dim = options.hidden_dim

        self.permutation_indices = [] if permutation_indices is None else permutation_indices
        self.batch_size = options.batch_size
        self.features = options.hidden_dim
        self.degree = degree

        # Add any missing cycles to have a complete group
        self.permutation_indices = complete_indices(self.degree, self.permutation_indices)
        self.permutation_group = symmetry_group(self.permutation_indices)
        self.no_identity_permutations = [p for p in self.permutation_group if sorted(p) != p]
        self.batch_no_identity_permutations = [(0,) + tuple(e + 1 for e in p) for p in self.no_identity_permutations]

        self.weights_scale = torch.sqrt(torch.scalar_tensor(self.features)) ** self.degree
