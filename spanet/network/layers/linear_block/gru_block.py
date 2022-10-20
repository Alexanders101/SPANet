import torch
from torch import Tensor, nn

from spanet.network.layers.linear_block.activations import create_activation, create_dropout, create_residual_connection
from spanet.network.layers.linear_block.normalizations import create_normalization
from spanet.network.layers.linear_block.masking import create_masking
from spanet.options import Options


class GRUGate(nn.Module):
    def __init__(self, hidden_dim, gate_initialization: float = 2.0):
        super(GRUGate, self).__init__()

        self.linear_W_r = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear_U_r = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.linear_W_z = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear_U_z = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.linear_W_g = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear_U_g = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.gate_bias = nn.Parameter(torch.ones(hidden_dim) * gate_initialization)

    def forward(self, vectors: Tensor, residual: Tensor) -> Tensor:
        r = torch.sigmoid(self.linear_W_r(vectors) + self.linear_U_r(residual))
        z = torch.sigmoid(self.linear_W_z(vectors) + self.linear_U_z(residual) - self.gate_bias)
        h = torch.tanh(self.linear_W_g(vectors) + self.linear_U_g(r * residual))

        return (1 - z) * residual + z * h


class GRUBlock(nn.Module):
    __constants__ = ['input_dim', 'output_dim', 'skip_connection', 'hidden_dim']

    # noinspection SpellCheckingInspection
    def __init__(self, options: Options, input_dim: int, output_dim: int, skip_connection: bool = False):
        super(GRUBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.skip_connection = skip_connection
        self.hidden_dim = int(round(options.transformer_dim_scale * input_dim))

        # Create normalization layer for keeping values in good ranges.
        self.normalization = create_normalization(options.normalization, input_dim)

        # The primary linear layers applied before the gate
        self.linear_1 = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            create_activation(options.linear_activation, self.hidden_dim),
            create_dropout(options.dropout)
        )

        self.linear_2 = nn.Sequential(
            nn.Linear(self.hidden_dim, output_dim),
            create_activation(options.linear_activation, output_dim),
            create_dropout(options.dropout)
        )

        # GRU layer to gate and project back to output. This will also handle the
        # self.gru = GRUGate(output_dim, input_dim)
        self.gru = GRUGate(output_dim)

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
        timesteps, batch_size, input_dim = x.shape

        # -----------------------------------------------------------------------------
        # Apply normalization first for this type of linear block.
        # output: [T, B, I]
        # -----------------------------------------------------------------------------
        output = self.normalization(x, sequence_mask)

        # -----------------------------------------------------------------------------
        # Flatten the data and apply the basic matrix multiplication and non-linearity.
        # output: [T * B, I]
        # -----------------------------------------------------------------------------
        output = output.reshape(timesteps * batch_size, self.input_dim)

        # -----------------------------------------------------------------------------
        # Apply linear layer with expansion in the middle.
        # output: [T * B, O]
        # -----------------------------------------------------------------------------
        output = self.linear_1(output)
        output = self.linear_2(output)

        # --------------------------------------------------------------------------
        # Reshape the data back into the time-series and apply normalization.
        # output: [T, B, O]
        # --------------------------------------------------------------------------
        output = output.reshape(timesteps, batch_size, self.output_dim)

        # -----------------------------------------------------------------------------
        # Apply gating mechanism and skip connection using the GRU mechanism.
        # output: [T, B, O]
        # -----------------------------------------------------------------------------
        if self.skip_connection:
            output = self.gru(output, self.residual(x))

        return self.masking(output, sequence_mask)
