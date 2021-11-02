from typing import Tuple

import torch
from torch import Tensor, nn, jit

from spanet.options import Options
from spanet.network.layers.jet_embedding import JetEmbedding


class JetEncoder(nn.Module):

    def __init__(self,
                 options: Options,
                 transformer_options: Tuple[int, int, int, float, str]):
        super(JetEncoder, self).__init__()

        encoder_layer = self.create_encoder_layer(transformer_options)
        self.encoder = nn.TransformerEncoder(encoder_layer, options.num_encoder_layers)

    @staticmethod
    def create_encoder_layer(transformer_options: Tuple[int, int, int, float, str]) -> nn.TransformerEncoderLayer:
        """ Encoder layer that will be given to nn.TransformerEncoder. This might be overwritten by subclasses. """
        return nn.TransformerEncoderLayer(*transformer_options)

    def forward(self, hidden: Tensor, padding_mask: Tensor, sequence_mask: Tensor) -> Tensor:
        """ Transform 4-momentum vectors into embedded and encoded representation for all branches.

        Parameters
        ----------
        hidden: [T, B, D]
            Combined activations from all inputs after embedding.
        padding_mask: [B, T]
            Negative mask indicating that a jet is padding for transformer.
        sequence_mask: [T, B, 1]
            Positive mask indicating jet is real.

        Returns
        -------
         hidden: [T, B, D]
            Hidden activations after central encoder.
        """

        # ----------------------------
        # Primary central transformer.
        # hidden: [T, B, D]
        # ----------------------------
        return self.encoder(hidden, src_key_padding_mask=padding_mask) * sequence_mask
