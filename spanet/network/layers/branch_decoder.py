from typing import Tuple, List
from opt_einsum import contract_expression

import torch
from torch import Tensor, nn, jit

from spanet.options import Options
from spanet.network.utilities import masked_log_softmax
from spanet.network.layers.stacked_encoder import StackedEncoder
from spanet.network.layers.branch_classifier import BranchClassifier
from spanet.network.symmetric_attention import SymmetricAttentionSplit, SymmetricAttentionFull


class BranchDecoder(nn.Module):
    # noinspection SpellCheckingInspection
    WEIGHTS_INDEX_NAMES = "ijklmn"
    DEFAULT_JET_COUNT = 16

    def __init__(self,
                 options: Options,
                 order: int,
                 permutation_indices: List[Tuple[int, ...]],
                 transformer_options: Tuple[int, int, int, float, str],
                 softmax_output: bool = True):
        super(BranchDecoder, self).__init__()

        self.order = order
        self.softmax_output = softmax_output
        self.combinatorial_scale = options.combinatorial_scale

        # Each branch has a personal encoder stack to extract particle-level data
        self.encoder = StackedEncoder(options,
                                      options.num_branch_embedding_layers,
                                      options.num_branch_encoder_layers,
                                      transformer_options)

        # Symmetric attention to create the output distribution
        attention_layer = SymmetricAttentionSplit if options.split_symmetric_attention else SymmetricAttentionFull
        self.attention = attention_layer(options, order, transformer_options, permutation_indices)

        # Optional output predicting if the particle was present or not
        self.classifier = BranchClassifier(options)

        self.num_targets = len(self.attention.permutation_group)
        self.permutation_indices = self.attention.permutation_indices

        self.padding_mask_operation = self.create_padding_mask_operation(options.batch_size)
        self.diagonal_mask_operation = self.create_diagonal_mask_operation()
        self.diagonal_masks = {}

    def create_padding_mask_operation(self, batch_size: int):
        weights_index_names = self.WEIGHTS_INDEX_NAMES[:self.order]
        operands = ','.join(map(lambda x: 'b' + x, weights_index_names))
        expression = f"{operands}->b{weights_index_names}"
        return expression

    def create_diagonal_mask_operation(self):
        weights_index_names = self.WEIGHTS_INDEX_NAMES[:self.order]
        operands = ','.join(map(lambda x: 'b' + x, weights_index_names))
        expression = f"{operands}->{weights_index_names}"
        return expression

    def create_output_mask(self, output: Tensor, sequence_mask: Tensor) -> Tensor:
        num_jets = output.shape[1]

        # batch_sequence_mask: [B, T, 1] Positive mask indicating jet is real.
        batch_sequence_mask = sequence_mask.transpose(0, 1).contiguous()

        # =========================================================================================
        # Padding mask
        # =========================================================================================
        padding_mask_operands = [batch_sequence_mask.squeeze() * 1] * self.order
        padding_mask = torch.einsum(self.padding_mask_operation, *padding_mask_operands)

        # =========================================================================================
        # Diagonal mask
        # =========================================================================================
        try:
            diagonal_mask = self.diagonal_masks[(num_jets, output.device)]
        except KeyError:
            identity = 1 - torch.eye(num_jets)
            identity = identity.type_as(output)

            diagonal_mask_operands = [identity * 1] * self.order
            diagonal_mask = torch.einsum(self.diagonal_mask_operation, *diagonal_mask_operands)
            diagonal_mask = diagonal_mask.unsqueeze(0) < (num_jets + 1 - self.order)
            self.diagonal_masks[(num_jets, output.device)] = diagonal_mask

        return (padding_mask * diagonal_mask).bool()

    def forward(self, x: Tensor, padding_mask: Tensor, sequence_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """ Create a distribution over jets for a given particle and a probability of its existence.

        Parameters
        ----------
        x : [T, B, D]
            Hidden activations after central encoder.
        padding_mask : [B, T]
            Negative mask for transformer input.
        sequence_mask : [T, B, 1]
            Positive mask for zeroing out padded vectors between operations.

        Returns
        -------
        output : [T, T, ...]
            Jet distribution for this particle.
        classification: [B]
            Probability of this particle existing in the data.
        """

        # ------------------------------------------------------
        # Apply the branch's independent encoder to each vector.
        # q : [T, B, D]
        # ------------------------------------------------------
        q = self.encoder(x, padding_mask, sequence_mask)

        # -----------------------------------------------
        # Run the encoded vectors through the classifier.
        # classification: [B]
        # -----------------------------------------------
        classification = self.classifier(q, sequence_mask)

        # -----------------------------------------------------------------
        # Create the jet distribution logits and the correctly shaped mask.
        # output : [T, T, ...]
        # mask : [T, T, ...]
        # -----------------------------------------------------------------
        output = self.attention(q, padding_mask, sequence_mask)
        mask = self.create_output_mask(output, sequence_mask)

        # ---------------------------------------------------------------------------
        # Need to reshape output to make softmax-calculation easier.
        # We transform the mask and output into a flat representation.
        # Afterwards, we apply a masked log-softmax to create the final distribution.
        # output : [T, T, ...]
        # mask : [T, T, ...]
        # ---------------------------------------------------------------------------
        if self.softmax_output:
            original_shape = output.shape
            batch_size = original_shape[0]

            output = output.reshape(batch_size, -1)
            mask = mask.reshape(batch_size, -1)

            output = masked_log_softmax(output, mask)
            output = output.view(*original_shape)

            # mask = mask.view(*original_shape)
            # offset = torch.log(mask.sum((1, 2, 3), keepdims=True).float()) * self.combinatorial_scale
            # output = output + offset

        return output, classification, mask
