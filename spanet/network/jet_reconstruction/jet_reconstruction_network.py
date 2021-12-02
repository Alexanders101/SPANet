from typing import Tuple, Dict, List

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

    def forward(
            self,
            sources: Tuple[Tuple[Tensor, Tensor], ...]
    ) -> Tuple[List[Tensor], List[Tensor], Dict[str, Tensor]]:
        # Embed all of the different input regression_vectors into the same latent space.
        embeddings, padding_masks, sequence_masks, global_masks = self.embedding(sources)

        # Extract features from data using transformer
        hidden, event_vector = self.encoder(embeddings, padding_masks, sequence_masks)

        # Create output lists for each particle in event.
        assignments = []
        detections = []

        regression_vectors = {
            "EVENT": event_vector
        }

        for decoder in self.decoders:
            (
                assignment,
                detection,
                assignment_mask,
                particle_vector,
                daughter_vectors
            ) = decoder(hidden, padding_masks, sequence_masks, global_masks)

            assignments.append(assignment)
            detections.append(detection)

            # Assign the summarising vectors to their correct structure
            regression_vectors["/".join([decoder.name, "PARTICLE"])] = particle_vector
            for daughter_name, daughter_vector in zip(decoder.daughter_names, daughter_vectors):
                regression_vectors["/".join([decoder.name, daughter_name])] = daughter_vector

        # Predict the valid regressions for any real values associated with the event
        regressions = self.regression_decoder(regression_vectors)

        # Pass the shared hidden state to every decoder branch
        return assignments, detections, regressions

    def predict(
            self,
            sources: Tuple[Tuple[Tensor, Tensor], ...]
    ) -> Tuple[TArray, TArray, Dict[str, TArray]]:

        with torch.no_grad():
            assignments, detections, regressions = self.forward(sources)

            # Extract assignment probabilities and find the least conflicting assignment.
            assignments = extract_predictions([
                np.nan_to_num(assignment.detach().cpu().numpy(), -np.inf)
                for assignment in assignments
            ])

            # Convert detection logits into probabilities and move to CPU.
            detections = np.stack([
                torch.sigmoid(detection).cpu().numpy()
                for detection in detections
            ])

            # Move regressions to CPU and away from torch.
            regressions = {
                key: value.cpu().numpy()
                for key, value in regressions.items()
            }

        return assignments, detections, regressions

    def predict_assignments(self, sources: Tuple[Tuple[Tensor, Tensor], ...]) -> np.ndarray:
        # Run the base prediction step
        with torch.no_grad():
            assignments = [
                np.nan_to_num(assignment.detach().cpu().numpy(), -np.inf)
                for assignment in self.forward(sources)[0]
            ]

        # Find the optimal selection of jets from the output distributions.
        return extract_predictions(assignments)

    def predict_assignments_and_detections(self, sources: Tuple[Tuple[Tensor, Tensor], ...]) -> Tuple[TArray, TArray]:
        assignments, detections, regressions = self.predict(sources)

        # Always predict the particle exists if we didn't train on it
        if self.options.classification_loss_scale == 0:
            detections += 1

        return assignments, detections >= 0.5
