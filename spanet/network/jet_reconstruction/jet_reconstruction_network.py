from typing import Tuple, Dict

import numpy as np
import torch
from torch import Tensor, nn

from spanet.network.jet_reconstruction.jet_reconstruction_base import JetReconstructionBase
from spanet.network.layers.regression_decoder import RegressionDecoder
from spanet.network.prediction_selection import extract_predictions
from spanet.network.layers.branch_decoder import BranchDecoder
from spanet.network.layers.embedding import CombinedVectorEmbedding
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

        self.embedding = CombinedVectorEmbedding(
            options,
            self.training_dataset.event_info,
            self.training_dataset
        )

        self.encoder = JetEncoder(
            options,
            transformer_options
        )

        self.decoders = nn.ModuleList([
            BranchDecoder(
                options,
                name,
                self.training_dataset.event_info.targets[name][0],
                size,
                permutation_indices,
                transformer_options,
                self.enable_softmax
            )
            for name, (size, permutation_indices) in self.training_dataset.target_symmetries
        ])

        self.regression_decoder = RegressionDecoder(
            options,
            self.training_dataset
        )

        # An example input for generating the network's graph, batch size of 2
        # self.example_input_array = tuple(x.contiguous() for x in self.training_dataset[:2][0])

    @property
    def enable_softmax(self):
        return True

    def forward(self, sources: Tuple[Tuple[Tensor, Tensor], ...]) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        # Embed all of the different input vectors into the same latent space.
        embeddings, padding_masks, sequence_masks, global_masks = self.embedding(sources)

        # Extract features from data using transformer
        hidden, event_vector = self.encoder(embeddings, padding_masks, sequence_masks)

        assignments = []
        presences = []

        vectors = {
            "EVENT": event_vector
        }

        for decoder in self.decoders:
            particle = decoder(hidden, padding_masks, sequence_masks, global_masks)
            selection, presence, selection_mask, particle_vector, daughter_vectors = particle

            assignments.append(selection)
            presences.append(presence)

            vectors["/".join([decoder.name, "PARTICLE"])] = particle_vector

            for daughter_name, daughter_vector in zip(decoder.daughter_names, daughter_vectors):
                vectors["/".join([decoder.name, daughter_name])] = daughter_vector

        # Pass the shared hidden state to every decoder branch
        return assignments, presences, self.regression_decoder(vectors)

    def predict_jets(self, sources: Tuple[Tuple[Tensor, Tensor], ...]) -> np.ndarray:
        # Run the base prediction step
        with torch.no_grad():
            predictions = []
            for prediction in self.forward(sources)[0]:
                prediction[torch.isnan(prediction)] = -np.inf
                predictions.append(prediction)

            # Find the optimal selection of jets from the output distributions.
            return extract_predictions(predictions)

    def predict_jets_and_particle_scores(self, sources: Tuple[Tuple[Tensor, Tensor], ...]) -> Tuple[TArray, TArray]:
        with torch.no_grad():
            predictions, classifications, regressions = self.forward(sources)

            clean_predictions = []
            scores = []

            for prediction, classification in zip(predictions, classifications):
                prediction[torch.isnan(prediction)] = -np.inf
                clean_predictions.append(prediction)

                scores.append(torch.sigmoid(classification).cpu().numpy())

            return extract_predictions(clean_predictions), np.stack(scores), regressions

    def predict_jets_and_particles(self, sources: Tuple[Tuple[Tensor, Tensor], ...]) -> Tuple[TArray, TArray]:
        predictions, scores = self.predict_jets_and_particle_scores(sources)

        # Always predict the particle exists if we didn't train on it
        if self.options.classification_loss_scale == 0:
            scores += 1

        return predictions, scores >= 0.5
