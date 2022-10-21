from torch import Tensor, nn

from spanet.network.layers.linear_block.activations import create_activation, create_dropout, create_residual_connection
from spanet.network.layers.linear_block.normalizations import create_normalization
from spanet.network.layers.linear_block.masking import create_masking
from spanet.options import Options


class BasicBlock(nn.Module):
    __constants__ = ['output_dim', 'skip_connection']

    # noinspection SpellCheckingInspection
    def __init__(self, options: Options, input_dim: int, output_dim: int, skip_connection: bool = False):
        super(BasicBlock, self).__init__()

        self.output_dim: int = output_dim
        self.skip_connection: bool = skip_connection

        # Basic matrix multiplication layer as the base.
        self.linear = nn.Linear(input_dim, output_dim)

        # Select non-linearity.
        self.activation = create_activation(options.linear_activation, output_dim)

        # Create normalization layer for keeping values in good ranges.
        self.normalization = create_normalization(options.normalization, output_dim)

        # Optional dropout for regularization.
        self.dropout = create_dropout(options.dropout)

        # Possibly need a linear layer to create residual connection.
        self.residual = create_residual_connection(skip_connection, input_dim, output_dim)

        # Mask out padding values
        self.masking = create_masking(options.masking)

    def forward(self, x: Tensor, sequence_mask: Tensor) -> Tensor:
        """ Simple robust linear layer with non-linearity, normalization, and dropout.

        Parameters
        ----------
        x: [T, B, D]
            Input data.
        sequence_mask: [T, B, 1]
            Positive mask indicating if the jet is a true jet or not.

        Returns
        -------
        y: [T, B, D]
            Output data.
        """
        max_jets, batch_size, dimensions = x.shape

        # -----------------------------------------------------------------------------
        # Flatten the data and apply the basic matrix multiplication and non-linearity.
        # x: [T * B, D]
        # y: [T * B, D]
        # -----------------------------------------------------------------------------
        x = x.reshape(max_jets * batch_size, dimensions)
        y = self.linear(x)
        y = self.activation(y)

        # ----------------------------------------------------------------------------
        # Optionally add a skip-connection to the network to add residual information.
        # y: [T * B, D]
        # ----------------------------------------------------------------------------
        if self.skip_connection:
            y = y + self.residual(x)

        # --------------------------------------------------------------------------
        # Reshape the data back into the time-series and apply regularization layers.
        # y: [T, B, D]
        # --------------------------------------------------------------------------
        y = y.reshape(max_jets, batch_size, self.output_dim)
        y = self.normalization(y, sequence_mask)
        return self.masking(self.dropout(y), sequence_mask)
