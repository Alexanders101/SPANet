from typing import Dict
from collections import OrderedDict

from torch import Tensor, nn

from spanet.options import Options
from spanet.dataset.regressions import regression_class
from spanet.network.layers.branch_linear import NormalizedBranchLinear
from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset


class RegressionDecoder(nn.Module):
    __constants__ = ['hidden_dim', 'num_layers']

    def __init__(self, options: Options, training_dataset: JetReconstructionDataset):
        super(RegressionDecoder, self).__init__()

        # Compute training dataset statistics to fix the final weight and bias.
        means, stds = training_dataset.compute_regression_statistics()

        # A unique linear decoder for each possible regression.
        # TODO make these non-unique for symmetric indices.
        networks = OrderedDict()
        for name, data in training_dataset.regressions.items():
            if data is None:
                continue

            networks[name] = NormalizedBranchLinear(
                options,
                options.num_regression_layers,
                regression_class(training_dataset.regression_types[name]),
                means[name],
                stds[name]
            )

        self.networks = nn.ModuleDict(networks)

    def forward(self, vectors: Dict[str, Tensor]) -> Dict[str, Tensor]:

        # vectors: Dict with mapping name -> [B, D]
        # outputs: Dict with mapping name -> [B, O_name]

        return {
            key: network(vectors['/'.join(key.split('/')[:-1])]).view(-1)
            for key, network in self.networks.items()
        }
