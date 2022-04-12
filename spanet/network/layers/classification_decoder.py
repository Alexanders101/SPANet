from typing import Dict, List
from collections import OrderedDict

from torch import Tensor, nn

from spanet.options import Options
from spanet.network.layers.branch_linear import MultiOutputBranchLinear, BranchLinear
from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset


class ClassificationDecoder(nn.Module):
    __constants__ = ['hidden_dim', 'num_layers']

    def __init__(self, options: Options, training_dataset: JetReconstructionDataset):
        super(ClassificationDecoder, self).__init__()

        # Compute training dataset statistics to fix the final weight and bias.
        counts = training_dataset.compute_classification_class_counts()

        # A unique linear decoder for each possible regression.
        networks = OrderedDict()
        for name, data in training_dataset.classifications.items():
            if data is None:
                continue

            networks[name] = BranchLinear(
                options,
                options.num_classification_layers,
                counts[name]
            )

            # networks[name] = MultiOutputBranchLinear(
            #     options,
            #     options.num_classification_layers,
            #     counts[name]
            # )

        self.networks = nn.ModuleDict(networks)

    def forward(self, vectors: Dict[str, Tensor]) -> Dict[str, List[Tensor]]:
        # vectors: Dict with mapping name -> [B, D]
        # outputs: Dict with mapping name -> [B, O_name]

        return {
            key: network(vectors['/'.join(key.split('/')[:-1])])
            for key, network in self.networks.items()
        }
