from torch import Tensor, nn, jit

from spanet.options import Options
from spanet.network.layers.masked_batch_norm import MaskedBatchNorm1D


class BatchNorm(nn.Module):
    __constants__ = ['output_dim']

    def __init__(self, output_dim: int):
        super(BatchNorm, self).__init__()

        self.output_dim = output_dim
        self.normalization = nn.BatchNorm1d(output_dim)

    # noinspection PyUnusedLocal
    def forward(self, x: Tensor, sequence_mask: Tensor) -> Tensor:
        max_jets, batch_size, output_dim = x.shape

        y = x.reshape(max_jets * batch_size, output_dim)
        y = self.normalization(y)
        return y.reshape(max_jets, batch_size, output_dim)


class MaskedBatchNorm(nn.Module):
    __constants__ = ['output_dim']

    def __init__(self, output_dim: int):
        super(MaskedBatchNorm, self).__init__()

        self.output_dim = output_dim
        self.normalization = MaskedBatchNorm1D(output_dim)

    # noinspection PyUnusedLocal
    def forward(self, x: Tensor, sequence_mask: Tensor) -> Tensor:
        max_jets, batch_size, output_dim = x.shape

        y = x.reshape(max_jets * batch_size, output_dim, 1)
        mask = sequence_mask.reshape(max_jets * batch_size, 1, 1)

        y = self.normalization(y, mask)

        return y.reshape(max_jets, batch_size, output_dim)


class LayerNorm(nn.Module):
    __constants__ = ['output_dim']

    def __init__(self, output_dim: int):
        super(LayerNorm, self).__init__()

        self.output_dim = output_dim
        self.normalization = nn.LayerNorm(output_dim)

    # noinspection PyUnusedLocal
    def forward(self, x: Tensor, sequence_mask: Tensor) -> Tensor:
        return self.normalization(x)


class LinearBlock(nn.Module):
    __constants__ = ['output_dim', 'skip_connection']

    # noinspection SpellCheckingInspection
    def __init__(self, options: Options, input_dim: int, output_dim: int, skip_connection: bool = False):
        super(LinearBlock, self).__init__()

        self.output_dim: int = output_dim
        self.skip_connection: bool = skip_connection

        # Basic matrix multiplication layer as the base
        self.linear = nn.Linear(input_dim, output_dim)

        # Select non-linearity. Either parametric or regular ReLU
        if options.linear_prelu_activation:
            self.activation = nn.PReLU(output_dim)
        else:
            self.activation = nn.ReLU()

        # Optional activation normalization. Either batch or layer norm.
        if options.normalization.lower() == "batchnorm":
            self.normalization = BatchNorm(output_dim)
        elif options.normalization.lower() == "maskedbatchnorm":
            self.normalization = MaskedBatchNorm1D(output_dim)
        elif options.normalization.lower() == "layernorm":
            self.normalization = LayerNorm(output_dim)
        else:
            self.normalization = nn.Identity()

        # Optional dropout
        if options.dropout > 0.0:
            self.dropout = nn.Dropout(options.dropout)
        else:
            self.dropout = nn.Identity()

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
            y = y + x

        # --------------------------------------------------------------------------
        # Reshape the data back into the time-series and apply regularization layers.
        # y: [T, B, D]
        # --------------------------------------------------------------------------
        y = y.reshape(max_jets, batch_size, self.output_dim)
        y = self.normalization(y, sequence_mask)
        return self.dropout(y) * sequence_mask


class LinearStack(nn.Module):
    __constants__ = ['num_layers']

    def __init__(self, options: Options, num_layers: int, hidden_dim: int, skip_connection: bool = True):
        super(LinearStack, self).__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            LinearBlock(options, hidden_dim, hidden_dim, skip_connection) for _ in range(num_layers)
        ])

    def forward(self, x: Tensor, sequence_mask: Tensor) -> Tensor:
        """ A stack of identically structured linear blocks in sequential order.

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
        output = x

        for layer in self.layers:
            output = layer(output, sequence_mask)

        return output
