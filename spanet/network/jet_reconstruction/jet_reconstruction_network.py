from typing import Tuple, Dict, List

import numpy as np
import torch
from torch import Tensor, nn

from spanet.options import Options

from spanet.network.layers.vector_encoder import JetEncoder
from spanet.network.layers.branch_decoder import BranchDecoder
from spanet.network.layers.embedding import MultiInputVectorEmbedding
from spanet.network.layers.regression_decoder import RegressionDecoder
from spanet.network.layers.classification_decoder import ClassificationDecoder


from spanet.network.prediction_selection import extract_predictions
from spanet.network.jet_reconstruction.jet_reconstruction_base import JetReconstructionBase

TArray = np.ndarray


class JetReconstructionNetwork(JetReconstructionBase):
    def __init__(self, options: Options, torch_script: bool = False):
        """ Base class defining the SPANet architecture.

        Parameters
        ----------
        options: Options
            Global options for the entire network.
            See network.options.Options
        """
        super(JetReconstructionNetwork, self).__init__(options)

        compile_module = torch.jit.script if torch_script else lambda x: x

        self.hidden_dim = options.hidden_dim

        self.embedding = compile_module(MultiInputVectorEmbedding(
            options,
            self.training_dataset
        ))

        self.encoder = compile_module(JetEncoder(
            options,
        ))

        self.branch_decoders = nn.ModuleList([
            BranchDecoder(
                options,
                name,
                self.training_dataset.event_info.assignments[name][0],
                size,
                permutation_indices,
                self.enable_softmax
            )
            for name, (size, permutation_indices) in self.training_dataset.assignment_symmetries
        ])

        self.regression_decoder = compile_module(RegressionDecoder(
            options,
            self.training_dataset
        ))

        self.classification_decoder = compile_module(ClassificationDecoder(
            options,
            self.training_dataset
        ))

        # An example input for generating the network's graph, batch size of 2
        # self.example_input_array = tuple(x.contiguous() for x in self.training_dataset[:2][0])

    @property
    def enable_softmax(self):
        return True

    def forward(
            self,
            sources: Tuple[Tuple[Tensor, Tensor], ...]
    ) -> Tuple[List[Tensor], List[Tensor], Dict[str, Tensor], Dict[str, List[Tensor]]]:
        # Embed all of the different input regression_vectors into the same latent space.
        embeddings, padding_masks, sequence_masks, global_masks = self.embedding(sources)

        # Extract features from data using transformer
        hidden, event_vector = self.encoder(embeddings, padding_masks, sequence_masks)

        # Create output lists for each particle in event.
        assignments = []
        detections = []

        encoded_vectors = {
            "EVENT": event_vector
        }

        for decoder in self.branch_decoders:
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
            encoded_vectors["/".join([decoder.name, "PARTICLE"])] = particle_vector
            for daughter_name, daughter_vector in zip(decoder.daughter_names, daughter_vectors):
                encoded_vectors["/".join([decoder.name, daughter_name])] = daughter_vector

        # Predict the valid regressions for any real values associated with the event
        regressions = self.regression_decoder(encoded_vectors)
        classifications = self.classification_decoder(encoded_vectors)

        # Pass the shared hidden state to every decoder branch
        return assignments, detections, regressions, classifications

    def predict(
            self,
            sources: Tuple[Tuple[Tensor, Tensor], ...]
    ) -> Tuple[TArray, TArray, Dict[str, TArray], Dict[str, List[TArray]]]:

        with torch.no_grad():
            assignments, detections, regressions, classifications = self.forward(sources)

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

            classifications = {
                key: value.cpu().argmax(1).numpy()
                for key, value in classifications.items()
            }

        return assignments, detections, regressions, classifications

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
        assignments, detections, regressions, classifications = self.predict(sources)

        # Always predict the particle exists if we didn't train on it
        if self.options.detection_loss_scale == 0:
            detections += 1

        return assignments, detections >= 0.5
