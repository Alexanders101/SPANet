from typing import Tuple

from torch import Tensor, nn

from spanet.options import Options
from spanet.dataset.types import InputType
from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset

from spanet.network.layers.embedding.normalizer import Normalizer
from spanet.network.layers.embedding.position_embedding import PositionEmbedding
from spanet.network.layers.embedding.global_vector_embedding import GlobalVectorEmbedding
from spanet.network.layers.embedding.relative_vector_embedding import RelativeVectorEmbedding
from spanet.network.layers.embedding.sequential_vector_embedding import SequentialVectorEmbedding


class CombinedVectorEmbedding(nn.Module):
    __constants__ = ["num_input_features"]

    def __init__(
            self,
            options: Options,
            training_dataset: JetReconstructionDataset,
            input_name: str,
            input_type: str
    ):
        super(CombinedVectorEmbedding, self).__init__()

        self.num_input_features = training_dataset.event_info.num_features(input_name)

        self.vector_embeddings = self.embedding_class(input_type)(options, self.num_input_features)
        self.position_embedding = PositionEmbedding(options.position_embedding_dim)

        if options.normalize_features:
            mean, std = training_dataset.compute_source_statistics()
            self.normalizer = Normalizer(mean[input_name], std[input_name])
        else:
            self.normalizer = nn.Identity()

    @staticmethod
    def embedding_class(embedding_type):
        if embedding_type == InputType.Sequential:
            return SequentialVectorEmbedding
        elif embedding_type == InputType.Relative:
            return RelativeVectorEmbedding
        elif embedding_type == InputType.Global:
            return GlobalVectorEmbedding
        else:
            raise ValueError(f"Unknown Embedding Type: {embedding_type}")

    def forward(self, source_data: Tensor, source_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Normalize incoming vectors based on training statistics.
        source_data = self.normalizer(source_data, source_mask)

        # Embed each vector type into the same latent space.
        embeddings, padding_mask, sequence_mask, global_mask = self.vector_embeddings(source_data, source_mask)

        # Add position embedding for this input type.
        embeddings = self.position_embedding(embeddings)

        return embeddings, padding_mask, sequence_mask, global_mask


