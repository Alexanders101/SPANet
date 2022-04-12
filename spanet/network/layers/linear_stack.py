from torch import nn, Tensor

from spanet.options import Options
from spanet.network.layers.linear_block import create_linear_block


class LinearStack(nn.Module):
    def __init__(self, options: Options, num_layers: int, hidden_dim: int, skip_connection: bool = True):
        super(LinearStack, self).__init__()

        self.layers = nn.ModuleList([
            create_linear_block(options, hidden_dim, hidden_dim, skip_connection)
            for _ in range(num_layers)
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
        output: [T, B, D]
            Output data.
        """
        output = x

        for layer in self.layers:
            output = layer(output, sequence_mask)

        return output


# noinspection PyUnusedLocal, PyMethodMayBeStatic
class LinearIdentity(nn.Module):
    def forward(self, x: Tensor, sequence_mask: Tensor) -> Tensor:
        return x


def create_linear_stack(
        options: Options,
        num_layers: int,
        hidden_dim: int,
        skip_connection: bool = False
):
    if num_layers > 0:
        return LinearStack(options, num_layers, hidden_dim, skip_connection)
    else:
        return LinearIdentity()
