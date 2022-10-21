from torch import Tensor, nn

from spanet.network.layers.linear_block.activations import create_activation, create_dropout, create_residual_connection
from spanet.network.layers.linear_block.normalizations import create_normalization
from spanet.network.layers.linear_block.masking import create_masking
from spanet.options import Options


class GLU(nn.Module):
    def __init__(self, hidden_dim: int):
        super(GLU, self).__init__()

        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return self.sigmoid(self.linear_1(x)) * self.linear_2(x)


class GatedBlock(nn.Module):
    __constants__ = ['output_dim', 'skip_connection', 'hidden_dim']

    # noinspection SpellCheckingInspection
    def __init__(self, options: Options, input_dim: int, output_dim: int, skip_connection: bool = False):
        super(GatedBlock, self).__init__()

        self.output_dim = output_dim
        self.skip_connection = skip_connection
        self.hidden_dim = int(round(options.transformer_dim_scale * input_dim))

        # The two fundemental linear layers for the gated network.
        self.linear_1 = nn.Linear(self.hidden_dim, output_dim)
        self.linear_2 = nn.Linear(input_dim, self.hidden_dim)

        # Select non-linearity.
        self.activation = create_activation(options.linear_activation, self.hidden_dim)

        # Create normalization layer for keeping values in good ranges.
        self.normalization = create_normalization(options.normalization, output_dim)

        # Optional dropout for regularization.
        self.dropout = create_dropout(options.dropout)

        # Possibly need a linear layer to create residual connection.
        self.residual = create_residual_connection(skip_connection, input_dim, output_dim)

        self.gate = GLU(output_dim)

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
        # x: [T * B, I]
        # -----------------------------------------------------------------------------
        x = x.reshape(max_jets * batch_size, dimensions)

        # -----------------------------------------------------------------------------
        # Apply both linear layers with expansion in the middle.
        # eta_2: [T * B, H]
        # eta_1: [T * B, O]
        # -----------------------------------------------------------------------------
        eta_2 = self.activation(self.linear_2(x))
        eta_1 = self.linear_1(eta_2)

        # -----------------------------------------------------------------------------
        # Apply gating mechanism to possibly ignore this layer.
        # output: [T * B, O]
        # -----------------------------------------------------------------------------
        output = self.dropout(eta_1)
        output = self.gate(output)

        # ----------------------------------------------------------------------------
        # Optionally add a skip-connection to the network to add residual information.
        # output: [T * B, O]
        # ----------------------------------------------------------------------------
        if self.skip_connection:
            output = output + self.residual(x)

        # --------------------------------------------------------------------------
        # Reshape the data back into the time-series and apply normalization.
        # output: [T, B, O]
        # --------------------------------------------------------------------------
        output = output.reshape(max_jets, batch_size, self.output_dim)
        output = self.normalization(output, sequence_mask)
        return self.masking(output, sequence_mask)
