from typing import Tuple

import torch
from torch import Tensor, nn, jit

from spanet.options import Options
from spanet.network.layers.jet_embedding import JetEmbedding


class JetEncoder(nn.Module):

    def __init__(self,
                 options: Options,
                 input_dim: int,
                 transformer_options: Tuple[int, int, int, float, str]):
        super(JetEncoder, self).__init__()

        self.mask_sequence_vectors = options.mask_sequence_vectors
        self.embedding = JetEmbedding(options, input_dim)

        encoder_layer = self.create_encoder_layer(transformer_options)
        self.encoder = nn.TransformerEncoder(encoder_layer, options.num_encoder_layers)

    @staticmethod
    def create_encoder_layer(transformer_options: Tuple[int, int, int, float, str]) -> nn.TransformerEncoderLayer:
        """ Encoder layer that will be given to nn.TransformerEncoder. This might be overwritten by subclasses. """
        return nn.TransformerEncoderLayer(*transformer_options)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """ Transform 4-momentum vectors into embedded and encoded representation for all branches.

        Parameters
        ----------
        x : [B, T, D]
            Input jet data.
        mask : [B, T]
            Positive mask indicating that the jet is a real jet.

        Returns
        -------
        hidden: [T, B, D]
            Hidden activations after central encoder.
        padding_mask: [B, T]
            Negative mask indicating that a jet is padding for transformer.
        sequence_mask: [T, B, 1]
            Positive mask indicating jet is real.
        """
        batch_size, max_jets, input_dim = x.shape

        # ----------------------------------------------
        # Create an inverse mask for transformer layers.
        # padding_mask: [B, T]
        # ----------------------------------------------
        padding_mask = ~mask

        # -------------------------------------------------------------------------------------------------
        # Create a positive mask indicating jet is real. This is for zeroing vectors at intermediate steps.
        # Alternatively, replace it with all ones if we are not masking (basically never).
        # sequence_mask: [T, B, 1]
        # -------------------------------------------------------------------------------------------------
        sequence_mask = mask.view(batch_size, max_jets, 1).transpose(0, 1).contiguous()
        if not self.mask_sequence_vectors:
            sequence_mask = torch.ones_like(sequence_mask)

        # -------------------------------------------------------------
        # Reshape vector to have time axis first for transformer input.
        # x: [T, B, D]
        # -------------------------------------------------------------
        x = x.transpose(0, 1).contiguous()

        # --------------------------------------------------
        # Perform embedding on all of the vectors uniformly.
        # hidden: [T, B, D]
        # --------------------------------------------------
        hidden = self.embedding(x, sequence_mask)

        # ----------------------------
        # Primary central transformer.
        # hidden: [T, B, D]
        # ----------------------------
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask) * sequence_mask

        return hidden, padding_mask, sequence_mask
