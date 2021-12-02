from typing import Tuple

import torch
from torch import Tensor, nn

from spanet.options import Options


class JetEncoder(nn.Module):

    def __init__(self,
                 options: Options,
                 transformer_options: Tuple[int, int, int, float, str]):
        super(JetEncoder, self).__init__()

        encoder_layer = self.create_encoder_layer(transformer_options)
        self.encoder = nn.TransformerEncoder(encoder_layer, options.num_encoder_layers)
        self.event_vector = nn.Parameter(torch.randn(1, 1, options.hidden_dim))

    @staticmethod
    def create_encoder_layer(transformer_options: Tuple[int, int, int, float, str]) -> nn.TransformerEncoderLayer:
        """ Encoder layer that will be given to nn.TransformerEncoder. This might be overwritten by subclasses. """
        return nn.TransformerEncoderLayer(*transformer_options)

    def forward(self, hidden: Tensor, padding_mask: Tensor, sequence_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """ Transform embedded "jet" vectors into contextual encoded representation for all branches.

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
        num_vectors, batch_size, hidden_dim = hidden.shape

        # -----------------------------------------------------------------------------
        # Add a "particle vector" which will store particle level data.
        # event_vector: [1, B, D]
        # combined_vectors: [T + 1, B, D]
        # -----------------------------------------------------------------------------
        event_vector = self.event_vector.expand(1, batch_size, hidden_dim)
        combined_vectors = torch.cat((event_vector, hidden), dim=0)

        # -----------------------------------------------------------------------------
        # Also modify the padding mask to indicate that the particle vector is real.
        # particle_padding_mask: [B, 1]
        # combined_padding_mask: [B, T + 1]
        # -----------------------------------------------------------------------------
        event_padding_mask = padding_mask.new_zeros(batch_size, 1)
        combined_padding_mask = torch.cat((event_padding_mask, padding_mask), dim=1)

        # -----------------------------------------------------------------------------
        # Run all of the vectors through transformer encoder
        # combined_vectors: [T + 1, B, D]
        # particle_vector: [B, D]
        # vectors: [T, B, D]
        # -----------------------------------------------------------------------------
        combined_vectors = self.encoder(combined_vectors, src_key_padding_mask=combined_padding_mask)
        event_vector, vectors = combined_vectors[0], combined_vectors[1:]

        return vectors * sequence_mask, event_vector
