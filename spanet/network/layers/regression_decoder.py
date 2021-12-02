from typing import Dict
from collections import OrderedDict

import torch
from torch import Tensor, nn, jit

from spanet.network.layers.branch_classifier import BranchLinear
from spanet.options import Options
from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset


class RegressionDecoder(nn.Module):
    __constants__ = ['hidden_dim', 'num_layers']

    def __init__(self, options: Options, training_dataset: JetReconstructionDataset):
        super(RegressionDecoder, self).__init__()

        # Compute training dataset statistics to fix the final weight and bias.
        means, stds = training_dataset.compute_regression_statistics()

        self.means = nn.ParameterDict({
            key: nn.Parameter(value)
            for key, value in means.items()
        })

        self.stds = nn.ParameterDict({
            key: nn.Parameter(value)
            for key, value in stds.items()
        })

        # A unique linear decoder for each possible regression.
        # TODO make these non-unique for symmetric indices.
        self.networks = nn.ModuleDict(OrderedDict(
            (name, BranchLinear(options, options.num_branch_classification_layers, data.shape[1]))
            for name, data in training_dataset.regressions.items()
            if data is not None
        ))

    def forward(self, vectors: Dict[str, Tensor]) -> Dict[str, Tensor]:

        # vectors: Dict with mapping name -> [B, D]
        # outputs: Dict with mapping name -> [B, O_name]

        return {
            key: self.stds[key] * network(vectors[key]) + self.means[key]
            for key, network in self.networks.items()
        }
