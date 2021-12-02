import torch
from torch import Tensor, nn, jit

from spanet.network.layers.linear_block import LinearStack
from spanet.options import Options


class BranchLinear(nn.Module):
    __constants__ = ['hidden_dim', 'num_layers']

    def __init__(self, options: Options, num_layers: int, num_outputs: int = 1):
        super(BranchLinear, self).__init__()

        self.hidden_dim = options.hidden_dim
        self.num_layers = num_layers

        self.hidden_layers = LinearStack(options,
                                         self.num_layers,
                                         self.hidden_dim,
                                         options.skip_connections)

        self.output_layer = nn.Linear(options.hidden_dim, num_outputs)

    def forward(self, single_vector: Tensor) -> Tensor:
        """ Produce a single classification output for a sequence of vectors.

        Parameters
        ----------
        single_vector : [B, D]
            Hidden activations after central encoder.

        Returns
        -------
        classification: [B, O]
            Probability of this particle existing in the data.
        """
        batch_size, input_dim = single_vector.shape

        # -----------------------------------------------------------------------------
        # Convert our single vector into a sequence of length 1.
        # Mostly just to re-use previous code.
        # sequence_mask: [1, B, 1]
        # single_vector: [1, B, D]
        # -----------------------------------------------------------------------------
        sequence_mask = torch.ones(1, batch_size, 1, dtype=torch.bool, device=single_vector.device)
        single_vector = single_vector.view(1, batch_size, input_dim)

        # ------------------------------------------------------------
        # Run through the linear layer stack and output the result
        # classification : [B, O]
        # ------------------------------------------------------------
        hidden = self.hidden_layers(single_vector, sequence_mask).squeeze()
        classification = self.output_layer(hidden)

        return classification
