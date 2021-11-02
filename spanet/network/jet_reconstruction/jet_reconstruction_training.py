from typing import Tuple, Dict

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from spanet.options import Options
from spanet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork
from spanet.network.utilities.divergence_losses import jet_cross_entropy_loss, jensen_shannon_divergence


class JetReconstructionTraining(JetReconstructionNetwork):
    def __init__(self, options: Options):
        super(JetReconstructionTraining, self).__init__(options)

        self.log_clip = torch.log(10 * torch.scalar_tensor(torch.finfo(torch.float32).eps)).item()

    def particle_classification_loss(self, classification: Tensor, target_mask: Tensor) -> Tensor:
        loss = F.binary_cross_entropy_with_logits(classification, target_mask.float(), reduction='none')
        return self.options.classification_loss_scale * loss

    def negative_log_likelihood(self,
                                predictions: Tuple[Tensor, ...],
                                classifications: Tuple[Tensor, ...],
                                targets: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor]:
        # We are only going to look at a single prediction points on the distribution for more stable loss calculation
        # We multiply the softmax values by the size of the permutation group to make every target the same
        # regardless of the number of sub-jets in each target particle
        predictions = [prediction + torch.log(torch.scalar_tensor(decoder.num_targets))
                       for prediction, decoder in zip(predictions, self.decoders)]

        # Convert the targets into a numpy array of tensors so we can use fancy indexing from numpy
        targets = np.array(targets, dtype='O')

        # Compute the loss on every valid permutation of the targets
        # TODO think of a way to avoid this memory transfer but keep permutation indices synced with checkpoint
        losses = []
        for permutation in self.event_permutation_tensor.cpu().numpy():
            loss = tuple(jet_cross_entropy_loss(P, T, M, self.options.focal_gamma) +
                         self.particle_classification_loss(C, M)
                         for P, C, (T, M)
                         in zip(predictions, classifications, targets[permutation]))

            losses.append(torch.sum(torch.stack(loss), dim=0))

        losses = torch.stack(losses)

        # Squash the permutation losses into a single value.
        # Typically we just take the minimum, but it might
        # be interesting to examine other methods.
        combined_losses, index = losses.min(0)

        if self.options.combine_pair_loss.lower() == "mean":
            combined_losses = losses.mean(0)
            index = 0

        if self.options.combine_pair_loss.lower() == "softmin":
            weights = F.softmin(losses, 0)
            combined_losses = (weights * losses).sum(0)

        return combined_losses, index

    def symmetric_divergence(self, predictions: Tuple[Tensor, ...], masks: Tensor) -> Tensor:
        # Make sure the gradient doesnt go through the target variable, only the prediction variable
        # Very unstable to take the gradient through both the target and prediction.
        # Since transpositions are symmetric, this will still be a symmetric operation.
        exp_predictions = [torch.exp(prediction.detach()) for prediction in predictions]

        results = []

        for i, j in self.training_dataset.unordered_event_transpositions:
            div = jensen_shannon_divergence(exp_predictions[i], predictions[i], exp_predictions[j], predictions[j])
            results.append(div * masks[i] * masks[j])

        return sum(results) / len(self.training_dataset.unordered_event_transpositions)

    def training_step(self, batch: Tuple[Tuple[Tensor, Tensor], ...], batch_nb: int) -> Dict[str, Tensor]:
        sources, num_jets, targets = batch

        # ===================================================================================================
        # Network Forward Pass
        # ---------------------------------------------------------------------------------------------------
        predictions = self.forward(sources)

        # Extract individual prediction data
        classifications = tuple(prediction[1] for prediction in predictions)
        predictions = tuple(prediction[0] for prediction in predictions)

        # ===================================================================================================
        # Initial log-likelihood loss for classification task
        # ---------------------------------------------------------------------------------------------------
        total_loss, best_indices = self.negative_log_likelihood(predictions, classifications, targets)

        # Log the classification loss to tensorboard.
        with torch.no_grad():
            self.log("loss/nll_loss", total_loss.mean())
            if torch.isnan(total_loss).any():
                raise ValueError("NLL Loss has diverged.")

        # Construct the newly permuted masks based on the minimal permutation found during NLL loss.
        permutations = self.event_permutation_tensor[best_indices].T
        masks = torch.stack([target[1] for target in targets])
        masks = torch.gather(masks, 0, permutations)

        # ===================================================================================================
        # Auxiliary loss term to prevent distributions from collapsing into single output.
        # ---------------------------------------------------------------------------------------------------
        if self.options.kl_loss_scale > 0:
            # Compute the symmetric loss between all valid pairs of distributions.
            kl_loss = self.symmetric_divergence(predictions, masks)

            with torch.no_grad():
                self.log("loss/symmetric_loss", -kl_loss.mean())
                if torch.isnan(kl_loss).any():
                    raise ValueError("Symmetric KL Loss has diverged.")

            total_loss = total_loss - kl_loss * self.options.kl_loss_scale

        # ===================================================================================================
        # Balance the loss based on the distribution of various classes in the dataset.
        # ---------------------------------------------------------------------------------------------------

        # Balance based on the particles present - only used in partial event training
        if self.balance_particles:
            class_indices = (masks * self.particle_index_tensor.unsqueeze(1)).sum(0)
            class_weights = self.particle_weights_tensor[class_indices]
            total_loss = total_loss * class_weights

        # Balance based on the number of jets in this event
        if self.balance_jets:
            class_weights = self.jet_weights_tensor[num_jets]
            total_loss = total_loss * class_weights

        # ===================================================================================================
        # Combine and return the loss
        # ---------------------------------------------------------------------------------------------------

        # TODO Simple mean for speed
        total_loss = len(targets) * total_loss.sum() / masks.sum()
        # total_loss = total_loss.mean()

        self.log("loss/total_loss", total_loss)
        return total_loss
