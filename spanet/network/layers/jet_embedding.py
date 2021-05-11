from typing import List

from torch import Tensor, nn, jit

from spanet.network.layers.linear_block import LinearBlock
from spanet.options import Options


class JetEmbedding(nn.Module):
    def __init__(self, options: Options, input_dim: int):
        super(JetEmbedding, self).__init__()

        self.input_dim = input_dim
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

    def forward(self, x: Tensor, sequence_mask: Tensor) -> Tensor:
        """ A stack of linear blocks with each layer doubling the hidden dimension

        Parameters
        ----------
        x: [T, B, D_1]
            Input data.
        sequence_mask: [T, B, 1]
            Positive mask indicating if the jet is a true jet or not.

        Returns
        -------
        y: [T, B, D_2]
            Output data.
        """
        output = x

        for layer in self.layers:
            output = layer(output, sequence_mask)

        return output
