from typing import List
from torch import Tensor, nn

from spanet.options import Options
from spanet.network.layers.branch_linear import BranchLinear


class MultiOutputBranchLinear(nn.Module):
    __constants__ = ['hidden_dim', 'num_layers']

    def __init__(
            self,
            options: Options,
            num_layers: int,
            num_outputs: Tensor
    ):
        super(MultiOutputBranchLinear, self).__init__()

        self.hidden_dim = options.hidden_dim
        self.num_layers = num_layers

        self.shared_layers = BranchLinear(
            options,
            max(self.num_layers - 1, 1),
            self.hidden_dim,
            batch_norm=False
        )

        self.output_layers = nn.ModuleList(
            BranchLinear(options, 1, output_dim.item())
            for output_dim in num_outputs
        )

    def forward(self, vector: Tensor) -> List[Tensor]:
        """ Produce a single classification output for a sequence of vectors.

        Parameters
        ----------
        vector : [B, D]
            Hidden activations after central encoder.

        Returns
        -------
        classification: [B, O]
            Probability of this particle existing in the data.
        """
        vector = self.shared_layers(vector)

        return [
            output_layer(vector)
            for output_layer in self.output_layers
        ]
