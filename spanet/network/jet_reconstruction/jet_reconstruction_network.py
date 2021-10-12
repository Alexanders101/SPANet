from typing import Tuple

import numpy as np
import torch
from torch import Tensor, nn

from spanet.network.jet_reconstruction.jet_reconstruction_base import JetReconstructionBase
from spanet.network.prediction_selection import extract_predictions
from spanet.network.layers.branch_decoder import BranchDecoder
from spanet.network.layers.jet_encoder import JetEncoder
from spanet.options import Options

TArray = np.ndarray


class JetReconstructionNetwork(JetReconstructionBase):
    def __init__(self, options: Options):
        """ Base class defining the SPANet architecture.

        Parameters
        ----------
        options: Options
            Global options for the entire network.
            See network.options.Options
        """
        super(JetReconstructionNetwork, self).__init__(options)

        self.hidden_dim = options.hidden_dim

        # Shared options for all transformer layers
        transformer_options = (options.hidden_dim,
                               options.num_attention_heads,
                               options.hidden_dim,
                               options.dropout,
                               options.transformer_activation)

        self.encoder = JetEncoder(options, self.training_dataset.num_features, transformer_options)
        self.decoders = nn.ModuleList([
            BranchDecoder(options, size, permutation_indices, transformer_options, self.enable_softmax)
            for _, (size, permutation_indices) in self.training_dataset.target_symmetries
        ])

        # An example input for generating the network's graph, batch size of 2
        self.example_input_array = tuple(x.contiguous() for x in self.training_dataset[:2][0])

    @property
    def enable_softmax(self):
        return True

    def forward(self, source_data: Tensor, source_mask: Tensor) -> Tuple[Tuple[Tensor, Tensor], ...]:
        # Normalize incoming data
        source_data = (source_data - self.mean) / self.std
        source_data = source_mask.unsqueeze(2) * source_data

        # Extract features from data using transformer
        hidden, padding_mask, sequence_mask = self.encoder(source_data, source_mask)

        # Pass the shared hidden state to every decoder branch
        return tuple(decoder(hidden, padding_mask, sequence_mask) for decoder in self.decoders)

    def predict_jets(self, source_data: Tensor, source_mask: Tensor) -> np.ndarray:
        # Run the base prediction step
        with torch.no_grad():
            predictions = []
            for prediction, _, _ in self.forward(source_data, source_mask):
                prediction[torch.isnan(prediction)] = -np.inf
                predictions.append(prediction)

            # Find the optimal selection of jets from the output distributions.
            return extract_predictions(predictions)

    def predict_jets_and_particle_scores(self, source_data: Tensor, source_mask: Tensor) -> Tuple[TArray, TArray]:
        with torch.no_grad():
            predictions = []
            scores = []
            for prediction, classification, _ in self.forward(source_data, source_mask):
                prediction[torch.isnan(prediction)] = -np.inf
                predictions.append(prediction)

                scores.append(torch.sigmoid(classification).cpu().numpy())

            return extract_predictions(predictions), np.stack(scores)

    def predict_jets_and_particles(self, source_data: Tensor, source_mask: Tensor) -> Tuple[TArray, TArray]:
        predictions, scores = self.predict_jets_and_particle_scores(source_data, source_mask)

        # Always predict the particle exists if we didn't train on it
        if self.options.classification_loss_scale == 0:
            scores += 1

        return predictions, scores >= 0.5
