import torch
from torch import Tensor, nn

from spanet.options import Options
from spanet.network.layers.linear_stack import create_linear_stack


class BranchLinear(nn.Module):
    __constants__ = ['hidden_dim', 'num_layers']

    def __init__(
            self,
            options: Options,
            num_layers: int,
            num_outputs: int = 1,
            batch_norm: bool = True
    ):
        super(BranchLinear, self).__init__()

        self.hidden_dim = options.hidden_dim
        self.num_layers = num_layers

        self.hidden_layers = create_linear_stack(
            options,
            self.num_layers,
            self.hidden_dim,
            options.skip_connections
        )

        # TODO Play around with this normalization layer
        if batch_norm:
            self.output_norm = nn.BatchNorm1d(options.hidden_dim)
        else:
            self.output_norm = nn.Identity()

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

        # ---------------------------------------------------------------------------
        # Run through hidden layer stack first, and then take the first timestep out.
        # hidden : [B, H]
        # ----------------------------------------------------------------------------
        hidden = self.hidden_layers(single_vector, sequence_mask)
        hidden = hidden.view(batch_size, self.hidden_dim)

        # ------------------------------------------------------------
        # Run through the linear layer stack and output the result
        # classification : [B, O]
        # ------------------------------------------------------------
        classification = self.output_layer(self.output_norm(hidden))

        return classification
