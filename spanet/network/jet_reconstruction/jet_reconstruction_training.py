from typing import Tuple, Dict, List

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from spanet.options import Options
from spanet.dataset.jet_reconstruction_dataset import TBatch
from spanet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork
from spanet.network.utilities.divergence_losses import assignment_cross_entropy_loss, jensen_shannon_divergence


def numpy_tensor_array(tensor_list):
    output = np.empty(len(tensor_list), dtype=np.object)
    output[:] = tensor_list

    return output


class JetReconstructionTraining(JetReconstructionNetwork):
    def __init__(self, options: Options, torch_script: bool = False):
        super(JetReconstructionTraining, self).__init__(options, torch_script)

        self.log_clip = torch.log(10 * torch.scalar_tensor(torch.finfo(torch.float32).eps)).item()

        self.particle_names = list(self.training_dataset.event_info.assignments.keys())
        self.daughter_names = {
            particle: self.training_dataset.event_info.assignments[particle][0]
            for particle in self.particle_names
        }

    def assignment_loss(self, assignment: Tensor, detection: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        assignment_loss = assignment_cross_entropy_loss(assignment, target, mask, self.options.focal_gamma)
        detection_loss = F.binary_cross_entropy_with_logits(detection, mask.float(), reduction='none')

        return (
                self.options.assignment_loss_scale * assignment_loss +
                self.options.detection_loss_scale * detection_loss
        )

    def compute_symmetric_assignment_losses(self, assignments: List[Tensor], detections: List[Tensor], targets):
        symmetric_assignment_losses = []

        # TODO think of a way to avoid this memory transfer but keep permutation indices synced with checkpoint
        # Compute a separate loss term for every possible target permutation.
        for permutation in self.event_permutation_tensor.cpu().numpy():

            # Find the assignment loss for each particle in this permutation.
            particle_assignment_losses = tuple(
                self.assignment_loss(assignment, detection, target, mask)
                for assignment, detection, (target, mask)
                in zip(assignments, detections, targets[permutation])
            )

            # The loss for a single permutation is the sum of particle losses.
            total_assignment_loss = torch.stack(particle_assignment_losses).sum(dim=0)

            symmetric_assignment_losses.append(total_assignment_loss)

        return torch.stack(symmetric_assignment_losses)

    def combine_symmetric_assignment_losses(self, symmetric_losses: Tensor) -> Tuple[Tensor, Tensor]:
        # Default option is to find the minimum loss term of the symmetric options.
        # We also store which permutation we used to achieve that minimal loss.
        combined_loss, index = symmetric_losses.min(0)

        # Simple average of all losses as a baseline.
        if self.options.combine_pair_loss.lower() == "mean":
            combined_loss = symmetric_losses.mean(0)

        # Soft minimum function to smoothly fuse all loss function weighted by their size.
        if self.options.combine_pair_loss.lower() == "softmin":
            weights = F.softmin(symmetric_losses, 0)
            combined_loss = (weights * symmetric_losses).sum(0)

        return combined_loss, index

    def symmetric_assignment_loss(
        self,
        assignments: List[Tensor],
        detections: List[Tensor],
        targets: Tuple[Tuple[Tensor, Tensor], ...]
    ) -> Tuple[Tensor, Tensor]:
        # We are only going to look at a single prediction points on the distribution for more stable loss calculation
        # We multiply the softmax values by the size of the permutation group to make every target the same
        # regardless of the number of sub-jets in each target particle
        assignments = [prediction + torch.log(torch.scalar_tensor(decoder.num_targets))
                       for prediction, decoder in zip(assignments, self.branch_decoders)]

        # Convert the targets into a numpy array of tensors so we can use fancy indexing from numpy
        targets = numpy_tensor_array(targets)

        # Compute the loss on every valid permutation of the targets
        symmetric_losses = self.compute_symmetric_assignment_losses(assignments, detections, targets)

        # Squash the permutation losses into a single value.
        return self.combine_symmetric_assignment_losses(symmetric_losses)

    def symmetric_divergence_loss(self, predictions: List[Tensor], masks: Tensor) -> Tensor:
        divergence_loss = []

        for i, j in self.training_dataset.unordered_event_transpositions:
            # Symmetric divergence between these two distributions
            div = jensen_shannon_divergence(predictions[i], predictions[j])

            # ERF term for loss
            loss = torch.exp(-(div ** 2))
            loss = loss.masked_fill(~masks[i], 0.0)
            loss = loss.masked_fill(~masks[j], 0.0)

            divergence_loss.append(loss)

        return torch.stack(divergence_loss).mean(0)
        # return -1 * torch.stack(divergence_loss).sum(0) / len(self.training_dataset.unordered_event_transpositions)

    def add_kl_loss(self, total_loss: Tensor, assignments, masks, weights) -> Tensor:
        if len(self.training_dataset.unordered_event_transpositions) == 0:
            return total_loss

        # Compute the symmetric loss between all valid pairs of distributions.
        kl_loss = self.symmetric_divergence_loss(assignments, masks)
        kl_loss = (weights * kl_loss).sum() / masks.sum()

        with torch.no_grad():
            self.log("loss/symmetric_loss", kl_loss)
            if torch.isnan(kl_loss):
                raise ValueError("Symmetric KL Loss has diverged.")

        return total_loss + self.options.kl_loss_scale * kl_loss

    def add_regression_loss(self, total_loss: Tensor, predictions: Dict[str, Tensor], targets:  Dict[str, Tensor]) -> Tensor:
        regression_terms = []

        for key in targets:
            current_prediction = predictions[key]
            current_target = targets[key]

            current_loss = torch.mean((current_prediction - current_target) ** 2, dim=1)
            current_loss = torch.nanmean(current_loss)

            with torch.no_grad():
                self.log(f"loss/regression/{key}", current_loss)

            regression_terms.append(current_loss)

        if len(regression_terms) == 0:
            return total_loss

        else:
            regression_loss = torch.stack(regression_terms).mean()
            return total_loss + self.options.regression_loss_scale * regression_loss

    def add_classification_loss(
            self,
            total_loss: Tensor,
            predictions: Dict[str, Tensor],
            targets: Dict[str, Tensor]
    ) -> Tensor:
        classification_terms = []

        for key in targets:
            current_prediction = predictions[key]
            current_target = targets[key]

            weight = None if not self.classification_weights else self.classification_weights[key]
            current_loss = F.cross_entropy(
                current_prediction,
                current_target,
                ignore_index=-1,
                weight=weight
            )


            classification_terms.append(current_loss)

            with torch.no_grad():
                self.log(f"loss/classification/{key}", current_loss)

        if len(classification_terms) == 0:
            return total_loss

        else:
            classification_loss = torch.stack(classification_terms).mean()
            return total_loss + self.options.classification_loss_scale * classification_loss

    def training_step(self, batch: TBatch, batch_nb: int) -> Dict[str, Tensor]:
        sources, num_jets, targets, regression_targets, classification_targets = batch

        # ===================================================================================================
        # Network Forward Pass
        # ---------------------------------------------------------------------------------------------------
        assignments, detections, regressions, classifications = self.forward(sources)

        # ===================================================================================================
        # Initial log-likelihood loss for classification task
        # ---------------------------------------------------------------------------------------------------
        total_loss, best_indices = self.symmetric_assignment_loss(assignments, detections, targets)

        # Construct the newly permuted masks based on the minimal permutation found during NLL loss.
        permutations = self.event_permutation_tensor[best_indices].T
        masks = torch.stack([target[1] for target in targets])
        masks = torch.gather(masks, 0, permutations)

        # ===================================================================================================
        # Balance the loss based on the distribution of various classes in the dataset.
        # ---------------------------------------------------------------------------------------------------

        # Default unity weight on correct device.
        weights = torch.ones_like(total_loss)

        # Balance based on the particles present - only used in partial event training
        if self.balance_particles:
            class_indices = (masks * self.particle_index_tensor.unsqueeze(1)).sum(0)
            weights *= self.particle_weights_tensor[class_indices]

        # Balance based on the number of jets in this event
        if self.balance_jets:
            weights *= self.jet_weights_tensor[num_jets]

        # ===================================================================================================
        # Take weighted average of the primary reconstruction loss.
        # ---------------------------------------------------------------------------------------------------

        # Take the weighted average of the reconstruction loss.
        # Replace it with a gradient-free version in case we are not doing reconstruction.
        total_loss = (weights * total_loss).sum() / masks.sum()
        if self.options.assignment_loss_scale == 0:
            total_loss = torch.zeros_like(total_loss)

        # Log the classification loss to tensorboard.
        with torch.no_grad():
            self.log("loss/nll_loss", total_loss)

            if torch.isnan(total_loss):
                raise ValueError("NLL Loss has diverged.")

            if torch.isinf(total_loss):
                raise ValueError("Assignment targets contain a collision.")

        # ===================================================================================================
        # Auxiliary loss terms which are added to reconstruction loss for alternative targets.
        # ---------------------------------------------------------------------------------------------------
        if self.options.kl_loss_scale > 0:
            total_loss = self.add_kl_loss(total_loss, assignments, masks, weights)

        if self.options.regression_loss_scale > 0:
            total_loss = self.add_regression_loss(total_loss, regressions, regression_targets)

        if self.options.classification_loss_scale > 0:
            total_loss = self.add_classification_loss(total_loss, classifications, classification_targets)

        # ===================================================================================================
        # Combine and return the loss
        # ---------------------------------------------------------------------------------------------------
        self.log("loss/total_loss", total_loss)
        return total_loss
