from typing import Tuple, List

import torch
from torch import Tensor, nn

from spanet.options import Options
from spanet.dataset.event_info import EventInfo
from spanet.network.layers.linear_block import LinearBlock
from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset
from spanet.network.layers.embedding.global_vector_embedding import GlobalVectorEmbedding
from spanet.network.layers.embedding.sequential_vector_embedding import SequentialVectorEmbedding


class CombinedVectorEmbedding(nn.Module):
    def __init__(self, options: Options, event_info: EventInfo, training_dataset: JetReconstructionDataset):
        super(CombinedVectorEmbedding, self).__init__()

        self.input_names = event_info.input_names

        # Primary embedding blocks to convert each input type into an identically shaped vector
        self.embeddings = nn.ModuleDict({
            input_name: self.embedding_class(input_type)(options, event_info.num_features(input_name))
            for input_name, input_type in event_info.input_types.items()
        })

        # Additional position parameters to encode that all vectors of a particle type come from the same source.
        self.position_embedding = nn.ParameterDict({
            input_name: nn.Parameter(torch.randn(1, 1, options.position_embedding_dim))
            for input_name, input_type in event_info.input_types.items()
        })

        # A final embedding layer to convert the position encoded vectors into a unified vector space.
        self.final_embedding_layer = LinearBlock(
            options,
            options.position_embedding_dim + options.hidden_dim,
            options.hidden_dim
        )

        if options.normalize_features:
            # Normalize datasets using training dataset statistics
            self.mean, self.std = training_dataset.compute_statistics()
        else:
            # Otherwise use default dummy normalizers
            self.mean = {input_name: torch.scalar_tensor(0) for input_name in event_info.input_names}
            self.std = {input_name: torch.scalar_tensor(1) for input_name in event_info.input_names}

        self.mean = nn.ParameterDict({
            key: nn.Parameter(val, requires_grad=False)
            for key, val in self.mean.items()
        })

        self.std = nn.ParameterDict({
            key: nn.Parameter(val, requires_grad=False)
            for key, val in self.std.items()
        })

    @staticmethod
    def embedding_class(embedding_type):
        if embedding_type.lower() == "sequential":
            return SequentialVectorEmbedding
        elif embedding_type.lower() == "global":
            return GlobalVectorEmbedding
        else:
            raise ValueError(f"Unknown Embedding Type: {embedding_type}")

    def forward(self, sources: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """

        Parameters
        ----------
        sources: List[[B, T_i, D_i]]
            A list containing each input source vectors in batch-first form.

        Returns
        -------
        embeddings: [T, B, D]
            Complete embeddings groups together in order of their inputs and in the proper latent dimension.
        padding_mask: [B, T]
            Negative mask indicating that a jet is padding for transformer.
        sequence_mask: [T, B, 1]
            Positive mask indicating jet is real.
        global_mask: [T]
            Negative mask for indicating a sequential variable or a global variable.
        """
        embeddings = []
        padding_masks = []
        sequence_masks = []
        global_masks = []

        for input_name, (source_data, source_mask) in zip(self.input_names, sources):
            batch_size, num_vectors, current_input_dim = source_data.shape

            # Normalize incoming vectors based on training statistics.
            source_data = (source_data - self.mean[input_name]) / self.std[input_name]
            source_data = source_mask.unsqueeze(-1) * source_data

            # Embed each vector type into the same latent space.
            (
                current_embeddings,
                current_padding_mask,
                current_sequence_mask,
                current_global_mask
            ) = self.embeddings[input_name](source_data, source_mask)

            # Add position embedding for this input type.
            current_position_embedding = self.position_embedding[input_name].expand(num_vectors, batch_size, -1)
            current_embeddings = torch.cat((current_embeddings, current_position_embedding), dim=2)

            # Accumulate all vectors into a single sequence.
            embeddings.append(current_embeddings)
            padding_masks.append(current_padding_mask)
            sequence_masks.append(current_sequence_mask)
            global_masks.append(current_global_mask)

        embeddings = torch.cat(embeddings, dim=0)
        padding_masks = torch.cat(padding_masks, dim=1)
        sequence_masks = torch.cat(sequence_masks, dim=0)
        global_masks = torch.cat(global_masks, dim=0)

        embeddings = self.final_embedding_layer(embeddings, sequence_masks)

        return embeddings, padding_masks, sequence_masks, global_masks
