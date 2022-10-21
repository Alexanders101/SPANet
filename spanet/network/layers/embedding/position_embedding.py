import torch
from torch import nn, Tensor


class PositionEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super(PositionEmbedding, self).__init__()

        self.position_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim))

    def forward(self, current_embeddings: Tensor) -> Tensor:
        num_vectors, batch_size, input_dim = current_embeddings.shape

        position_embedding = self.position_embedding.expand(num_vectors, batch_size, -1)
        return torch.cat((current_embeddings, position_embedding), dim=2)
