from itertools import islice
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor, nn, jit

from spanet.options import Options
from spanet.network.layers.stacked_encoder import StackedEncoder
from spanet.network.symmetric_attention.symmetric_attention_base import SymmetricAttentionBase
from spanet.network.utilities.linear_form import create_symmetric_function


# noinspection SpellCheckingInspection
class SymmetricAttentionSplit(SymmetricAttentionBase):
    def __init__(self,
                 options: Options,
                 order: int,
                 transformer_options: Tuple[int, int, int, float, str] = None,
                 permutation_indices: List[Tuple[int, ...]] = None,
                 attention_dim: int = None) -> None:

        super(SymmetricAttentionSplit, self).__init__(options,
                                                      order,
                                                      transformer_options,
                                                      permutation_indices,
                                                      attention_dim)

        # Each potential jet gets its own encoder in order to extract information for attention.
        self.encoders = nn.ModuleList([
            StackedEncoder(options,
                           options.num_jet_embedding_layers,
                           options.num_jet_encoder_layers,
                           transformer_options)

            for _ in range(order)
        ])

        # After encoding, the jets are fed into a final linear layer to extract logits.
        self.linear_layers = nn.ModuleList([
            nn.Linear(options.hidden_dim, self.attention_dim)
            for _ in range(order)
        ])

        # Add additional non-linearity on top of the linear layer.
        self.activations = nn.ModuleList([
            nn.PReLU(self.attention_dim)
            for _ in range(order)
        ])

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
        operations = ','.join(islice(operations, self.order))

        result = f"->b{''.join(input_index_names[:self.order])}"

        return operations + result

    def forward(self, x: Tensor, padding_mask: Tensor, sequence_mask: Tensor) -> Tensor:
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
        num_jets, batch_size, features = x.shape

        # ---------------------------------------------------------
        # Construct the transformed attention vectors for each jet.
        # ys: [[T, B, D], ...]
        # ---------------------------------------------------------
        ys = []
        for encoder, linear_layer, activation in zip(self.encoders, self.linear_layers, self.activations):
            # ------------------------------------------------------
            # First pass the input through this jet's encoder stack.
            # y: [T, B, D]
            # ------------------------------------------------------
            y = encoder(x, padding_mask, sequence_mask)

            # --------------------------------------------------------
            # Flatten and apply the final linear layer to each vector.
            # y: [T, B, D]
            # ---------------------------------------------------------
            y = y.reshape(num_jets * batch_size, -1)
            y = linear_layer(y)
            y = activation(y)
            y = y.reshape(num_jets, batch_size, self.attention_dim) * sequence_mask

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

        return output
