from typing import Tuple
import numpy as np

import torch
from torch import Tensor, nn

from spanet.network.layers.embedding_stack import EmbeddingStack
from spanet.network.layers.linear_block import create_linear_block
from spanet.options import Options


class RelativeVectorEmbedding(nn.Module):
    """
    An implementation of a lorentz-invariant embeddings using attention.
    Inspired by Covariant Attention from Qiu Et al.

    A Holistic Approach to Predicting Top Quark Kinematic Properties with the Covariant Particle Transformer
    Shikai Qiu, Shuo Han, Xiangyang Ju, Benjamin Nachman, and Haichen Wang
    https://arxiv.org/pdf/2203.05687.pdf
    """
    __constants__ = ["input_dim", "mask_sequence_vectors", "attention_scale"]

    def __init__(self, options: Options, input_dim: int):
        super(RelativeVectorEmbedding, self).__init__()

        self.input_dim = input_dim
        self.mask_sequence_vectors = options.mask_sequence_vectors

        self.shared_embedding_stack = EmbeddingStack(options, input_dim)
        self.shared_embedding_norm = nn.LayerNorm(options.hidden_dim)

        self.query_embedding = nn.Linear(options.hidden_dim, options.hidden_dim, bias=False)
        self.key_embedding = nn.Linear(options.hidden_dim, options.hidden_dim, bias=False)
        self.value_embedding = nn.Linear(options.hidden_dim, options.hidden_dim, bias=False)
        self.attention_scale = np.sqrt(options.hidden_dim)

        self.output = create_linear_block(options, options.hidden_dim, options.hidden_dim, options.skip_connections)

    def forward(self, vectors: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """ A stack of linear blocks with each layer doubling the hidden dimension

        Parameters
        ----------
        vectors : [B, T, T, I]
            Relative vector data.
        mask : [B, T, T]
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
        batch_size, max_vectors, _, input_dim = vectors.shape

        # -----------------------------------------------
        # Get the full pairwise mask
        # square_mask: [T, T, B]
        # -----------------------------------------------
        square_max_vectors = max_vectors * max_vectors
        square_mask = mask.permute(1, 2, 0)

        # -----------------------------------------------
        # Construct output linear masks from diagonal of relative mask.
        # padding_mask: [B, T]
        # sequence_mask: [T, B, 1]
        # -----------------------------------------------
        sequence_mask = square_mask.diagonal()
        padding_mask = ~sequence_mask
        sequence_mask = sequence_mask.transpose(0, 1).contiguous().unsqueeze(-1)

        # -----------------------------------------------
        # Flatten square mask to allow input to the linear layers
        # square_mask: [T * T, B]
        # -----------------------------------------------
        square_mask = square_mask.reshape(square_max_vectors, batch_size).unsqueeze(-1)

        # --------------------------------------------------------------
        # Perform main embedding on vectors to get them up to hidde dim.
        # vectors: [T, T, B, D]
        # --------------------------------------------------------------
        vectors = vectors.reshape(batch_size, square_max_vectors, -1).permute(1, 0, 2)
        vectors = self.shared_embedding_stack(vectors, square_mask)
        vectors = self.shared_embedding_norm(vectors)
        vectors = vectors.reshape(max_vectors, max_vectors, batch_size, -1)

        # -----------------------------------------------
        # Compute attention components
        # keys: [T, T, B, D]
        # values: [T, T, B, D]
        # queries: [T, B, D]
        # -----------------------------------------------
        keys = self.key_embedding(vectors)
        values = self.value_embedding(vectors)
        queries = vectors.diagonal().permute(2, 0, 1)
        queries = self.query_embedding(queries)

        # ---------------------------------------------------
        # Attention mask for zero-ing out the masked vectors.
        # Ensure the diagonal is True to avoid nans.
        # attention_mask: [T, T, B]
        # ---------------------------------------------------
        attention_mask = square_mask.view(max_vectors, max_vectors, batch_size)
        attention_mask |= torch.eye(attention_mask.shape[0], dtype=attention_mask.dtype, device=attention_mask.device).unsqueeze(-1)

        # ---------------------------------------------------
        # Compute attention softmax weights using dot-product
        # attention_mask: [T, T, B]
        # ---------------------------------------------------
        attention_weights = torch.einsum("rbd,rcbd->rcb", queries, keys) / self.attention_scale
        attention_weights = attention_weights + torch.log(attention_mask)
        attention_weights = torch.softmax(attention_weights, dim=1)

        embeddings = attention_weights.unsqueeze(-1) * values
        embeddings = embeddings.sum(1)
        embeddings = self.output(embeddings, sequence_mask)

        # ----------------------------------------------------------------------------
        # Create a negative mask indicating that all of the vectors that we embed will
        # be sequential variables and not global variables.
        # global_mask: [T]
        # ----------------------------------------------------------------------------
        global_mask = sequence_mask.new_ones((max_vectors,))

        return embeddings, padding_mask, sequence_mask, global_mask
