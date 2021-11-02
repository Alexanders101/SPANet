from typing import List, Tuple

import torch
from torch import Tensor, nn, jit

from spanet.network.layers.linear_block import LinearBlock
from spanet.options import Options


class JetEmbedding(nn.Module):
    __constants__ = ["input_dim", "mask_sequence_vectors"]

    def __init__(self, options: Options, input_dim: int):
        super(JetEmbedding, self).__init__()

        self.input_dim = input_dim
        self.mask_sequence_vectors = options.mask_sequence_vectors
        self.layers = nn.ModuleList(self.create_embedding_layers(options, input_dim))

    @staticmethod
    def create_embedding_layers(options: Options, input_dim: int) -> List[LinearBlock]:
        """ Create a stack of linear layer with increasing hidden dimensions.

        Each hidden layer will have double the dimensions as the previous, beginning with the
        size of the feature-space and ending with the hidden_dim specified in options.
        """
        embedding_layers = [LinearBlock(options, input_dim, options.initial_embedding_dim)]
        current_embedding_dim = options.initial_embedding_dim

        for i in range(options.num_embedding_layers):
            next_embedding_dim = 2 * current_embedding_dim
            if next_embedding_dim >= options.hidden_dim:
                break

            embedding_layers.append(LinearBlock(options, current_embedding_dim, next_embedding_dim))
            current_embedding_dim = next_embedding_dim

        embedding_layers.append(LinearBlock(options, current_embedding_dim, options.hidden_dim))

        return embedding_layers

    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """ A stack of linear blocks with each layer doubling the hidden dimension

        Parameters
        ----------
        x : [B, T, D]
            Input jet data.
        mask : [B, T]
            Positive mask indicating that the jet is a real jet.

        Returns
        -------
        hidden: [T, B, D]
            Hidden activations after embedding.
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
        output = x.transpose(0, 1).contiguous()

        for layer in self.layers:
            output = layer(output, sequence_mask)

        return output, padding_mask, sequence_mask
