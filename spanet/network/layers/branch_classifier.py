from torch import Tensor, nn, jit

from spanet.network.layers.linear_block import LinearStack
from spanet.options import Options


class BranchClassifier(nn.Module):
    __constants__ = ['hidden_dim', 'num_layers']

    def __init__(self, options: Options):
        super(BranchClassifier, self).__init__()

        self.hidden_dim = options.hidden_dim
        self.num_layers = options.num_branch_classification_layers

        self.hidden_layers = LinearStack(options,
                                         self.num_layers,
                                         self.hidden_dim,
                                         options.skip_connections)

        self.output_layer = nn.Linear(options.hidden_dim, 1)

    def forward(self, q: Tensor, sequence_mask: Tensor) -> Tensor:
        """ Produce a single classification output for a sequence of vectors.

        Parameters
        ----------
        q : [T, B, D]
            Hidden activations after central encoder.
        sequence_mask : [T, B, 1]
            Positive mask for zeroing out padded vectors between operations.

        Returns
        -------
        classification: [B]
            Probability of this particle existing in the data.
        """

        # ------------------------------------------------------------
        # Collapse the sequence vectors into a single vector as a sum.
        # hidden_dim : [1, B, D]
        # sequence_mask : [1, B, 1]
        # ------------------------------------------------------------
        hidden = (q * sequence_mask).sum(0, keepdim=True) / sequence_mask.sum(0, keepdim=True)
        sequence_mask = sequence_mask.sum(dim=0, keepdim=True) > 0

        # ------------------------------------------------------------
        # Run through the linear layer stack and output the result
        # classification : [B]
        # ------------------------------------------------------------
        hidden = self.hidden_layers(hidden, sequence_mask).squeeze()
        classification = self.output_layer(hidden).squeeze()

        return classification
