from typing import Tuple

import torch
from torch import Tensor, nn

from spanet.network.layers.embedding_stack import EmbeddingStack
from spanet.options import Options


class SequentialVectorEmbedding(nn.Module):
    __constants__ = ["input_dim", "mask_sequence_vectors"]

    def __init__(self, options: Options, input_dim: int):
        super(SequentialVectorEmbedding, self).__init__()

        self.input_dim = input_dim
        self.mask_sequence_vectors = options.mask_sequence_vectors
        self.embedding_stack = EmbeddingStack(options, input_dim)

    def forward(self, vectors: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """ A stack of linear blocks with each layer doubling the hidden dimension

        Parameters
        ----------
        vectors : [B, T, I]
            Input vector data.
        mask : [B, T]
            Positive mask indicating that the jet is a real jet.

        Returns
        -------
        embeddings: [T, B, D]
            Hidden activations after embedding.
        padding_mask: [B, T]
            Negative mask indicating that a jet is padding for transformer.
        sequence_mask: [T, B, 1]
            Positive mask indicating jet is real.
        global_mask: [T]
            Negative mask for indicating a sequential variable or a global variable.
        """
        batch_size, max_vectors, input_dim = vectors.shape

        # -----------------------------------------------
        # Create an negative mask for transformer layers.
        # padding_mask: [B, T]
        # -----------------------------------------------
        padding_mask = ~mask

        # -------------------------------------------------------------------------------------------------
        # Create a positive mask indicating jet is real. This is for zeroing vectors at intermediate steps.
        # Alternatively, replace it with all ones if we are not masking (basically never).
        # sequence_mask: [T, B, 1]
        # -------------------------------------------------------------------------------------------------
        sequence_mask = mask.view(batch_size, max_vectors, 1).transpose(0, 1).contiguous()
        if not self.mask_sequence_vectors:
            sequence_mask = torch.ones_like(sequence_mask)

        # ----------------------------------------------------------------------------
        # Create a negative mask indicating that all of the vectors that we embed will
        # be sequential variables and not global variables.
        # global_mask: [T]
        # ----------------------------------------------------------------------------
        global_mask = sequence_mask.new_ones((max_vectors,))

        # -------------------------------------------------------------
        # Reshape vector to have time axis first for transformer input.
        # output: [T, B, I]
        # -------------------------------------------------------------
        embeddings = vectors.transpose(0, 1).contiguous()

        # --------------------------------
        # Embed vectors into latent space.
        # output: [T, B, D]
        # --------------------------------
        embeddings = self.embedding_stack(embeddings, sequence_mask)

        return embeddings, padding_mask, sequence_mask, global_mask
