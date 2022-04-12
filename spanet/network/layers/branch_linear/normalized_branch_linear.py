from torch import Tensor, nn

from spanet.options import Options
from spanet.network.layers.branch_linear import BranchLinear


class NormalizedBranchLinear(nn.Module):
    __constants__ = ['hidden_dim', 'num_layers', "num_outputs"]

    def __init__(
            self,
            options: Options,
            num_layers: int,
            mean: Tensor,
            std: Tensor
    ):
        super(NormalizedBranchLinear, self).__init__()

        self.hidden_dim = options.hidden_dim
        self.num_outputs = mean.shape[1]
        self.num_layers = num_layers

        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)
        self.linear = BranchLinear(
            options,
            self.num_layers,
            self.num_outputs
        )

    def forward(self, vector: Tensor) -> Tensor:
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
        return self.std * self.linear(vector) + self.mean
