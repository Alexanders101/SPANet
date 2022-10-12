from itertools import islice
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from spanet.options import Options
from spanet.network.layers.stacked_encoder import StackedEncoder
from spanet.network.symmetric_attention.symmetric_attention_base import SymmetricAttentionBase
from spanet.network.layers.linear_block.masking import create_masking
from spanet.network.utilities.linear_form import create_symmetric_function


# noinspection SpellCheckingInspection
class SymmetricAttentionSplit(SymmetricAttentionBase):
    def __init__(self,
                 options: Options,
                 degree: int,
                 permutation_indices: List[Tuple[int, ...]] = None,
                 attention_dim: int = None) -> None:

        super(SymmetricAttentionSplit, self).__init__(
            options,
            degree,
            permutation_indices,
            attention_dim
        )

        # Each potential jet gets its own encoder in order to extract information for attention.
        self.encoders = nn.ModuleList([
            StackedEncoder(
                options,
                options.num_jet_embedding_layers,
                options.num_jet_encoder_layers
            )
            for _ in range(degree)
        ])

        # After encoding, the jets are fed into a final linear layer to extract logits.
        # TODO Play around with bias
        self.linear_layers = nn.ModuleList([
            nn.Linear(options.hidden_dim, self.attention_dim, bias=True)
            for _ in range(degree)
        ])

        # Mask the vectors before applying attentino operation.
        self.masking = create_masking(options.masking)

        # This layer ensures symmetric output by symmetrizing the OUTPUT tensor.
        self.symmetrize_tensor = create_symmetric_function(self.batch_no_identity_permutations)

        # Operation to perform general n-dimensional attention.
        self.contraction_operation = self.make_contraction()

        self.reset_parameters()

    def reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.linear_layers.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_contraction(self):
        input_index_names = np.array(list(self.INPUT_INDEX_NAMES))

        operations = map(lambda x: f"{x}bi", input_index_names)
        operations = ','.join(islice(operations, self.degree))

        result = f"->b{''.join(input_index_names[:self.degree])}"

        return operations + result

    def forward(self, x: Tensor, padding_mask: Tensor, sequence_mask: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """ Perform symmetric attention on the hidden vectors and produce the output logits.

        This is the approximate version which learns embedding layers and computes a trivial linear form.

        Parameters
        ----------
        x : [T, B, D]
            Hidden activations after branch encoders.
        padding_mask: [B, T]
            Negative mask indicating that a jet is padding for transformer.
        sequence_mask: [T, B, 1]
            Positive mask indicating jet is real.

        Returns
        -------
        output : [T, T, ...]
            Prediction logits for this particle.
        """
        # ---------------------------------------------------------
        # Construct the transformed attention vectors for each jet.
        # ys: [[T, B, D], ...]
        # ---------------------------------------------------------
        ys = []
        daughter_vectors = []
        for encoder, linear_layer in zip(self.encoders, self.linear_layers):
            # ------------------------------------------------------
            # First pass the input through this jet's encoder stack.
            # y: [T, B, D]
            # ------------------------------------------------------
            y, daughter_vector = encoder(x, padding_mask, sequence_mask)

            # --------------------------------------------------------
            # Flatten and apply the final linear layer to each vector.
            # y: [T, B, D]
            # ---------------------------------------------------------
            y = linear_layer(y)
            y = self.masking(y, sequence_mask)

            # Accumulate vectors into stack for each daughter of this particle.
            daughter_vectors.append(daughter_vector)
            ys.append(y)

        # -------------------------------------------------------
        # Construct the output logits via general self-attention.
        # output: [T, T, ...]
        # -------------------------------------------------------
        output = torch.einsum(self.contraction_operation, *ys)
        output = output / self.weights_scale

        # ---------------------------------------------------
        # Symmetrize the output according to group structure.
        # output: [T, T, ...]
        # ---------------------------------------------------
        # TODO Perhaps make the encoder layers match in the symmetric dimensions.
        output = self.symmetrize_tensor(output)

        return output, daughter_vectors
