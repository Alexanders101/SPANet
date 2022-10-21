import h5py
import numpy as np

import torch

from spanet.dataset.types import SpecialKey, Statistics, Source
from spanet.dataset.inputs.BaseInput import BaseInput



class RelativeInput(BaseInput):
    # noinspection PyAttributeOutsideInit
    def load(self, hdf5_file: h5py.File, limit_index: np.ndarray):
        input_group = [SpecialKey.Inputs, self.input_name]

        # Load in the mask for this vector input
        invariant_mask = torch.from_numpy(self.dataset(hdf5_file, input_group, SpecialKey.Mask)[:]).contiguous()
        covariant_mask = invariant_mask[:, None, :] * invariant_mask[:, :, None]

        # Get all the features in this group, we need to figure out if each one is invariant or covariant
        feature_names = list(self.dataset(hdf5_file, [SpecialKey.Inputs], self.input_name).keys())
        feature_names.remove(SpecialKey.Mask)

        # Separate the two types of features by their shape
        self.invariant_features = []
        self.covariant_features = []

        for feature in self.event_info.input_features[self.input_name]:
            if len(self.dataset(hdf5_file, input_group, feature[0]).shape) == 2:
                self.invariant_features.append(feature)
            else:
                self.covariant_features.append(feature)

        # Load in vector features into a pre-made buffer
        num_jets = invariant_mask.shape[1]
        invariant_data = torch.empty(len(self.invariant_features), self.num_events, num_jets, dtype=torch.float32)
        covariant_data = torch.empty(len(self.covariant_features), self.num_events, num_jets, num_jets, dtype=torch.float32)

        invariant_index = 0
        covariant_index = 0

        for (feature, _, log_transform) in self.event_info.input_features[self.input_name]:
            current_dataset = self.dataset(hdf5_file, input_group, feature)
            if len(current_dataset.shape) == 2:
                current_data = invariant_data
                current_mask = invariant_mask
                current_index = invariant_index
                invariant_index += 1
            else:
                current_data = covariant_data
                current_mask = covariant_mask
                current_index = covariant_index
                covariant_index += 1

            current_dataset.read_direct(current_data[current_index].numpy())
            if log_transform:
                # torch.clamp_(current_data[current_index], min=1e-6)
                current_data[current_index] += 1
                torch.log_(current_data[current_index])
                current_data[current_index] *= current_mask

        # Reshape and limit data to the limiting index.
        invariant_data = invariant_data.permute(1, 2, 0)
        covariant_data = covariant_data.permute(1, 2, 3, 0)

        self.invariant_data = invariant_data[limit_index].contiguous()
        self.covariant_data = covariant_data[limit_index].contiguous()

        self.invariant_mask = invariant_mask[limit_index].contiguous()
        self.covariant_mask = covariant_mask[limit_index].contiguous()

    # noinspection PyAttributeOutsideInit
    def limit(self, event_mask):
        self.invariant_data = self.invariant_data[event_mask].contiguous()
        self.covariant_data = self.covariant_data[event_mask].contiguous()

        self.invariant_mask = self.invariant_mask[event_mask].contiguous()
        self.covariant_mask = self.covariant_mask[event_mask].contiguous()

    def compute_statistics(self) -> Statistics:
        masked_invariant_data = self.invariant_data[self.invariant_mask]
        masked_covariant_data = self.covariant_data[self.covariant_mask]

        masked_invariant_mean = masked_invariant_data.mean(0)
        masked_invariant_std = masked_invariant_data.std(0)
        masked_invariant_std[masked_invariant_std < 1e-5] = 1

        masked_covariant_mean = masked_covariant_data.mean(0)
        masked_covariant_std = masked_covariant_data.std(0)
        masked_covariant_std[masked_covariant_std < 1e-5] = 1

        unnormalized_invariant_features = ~np.array([feature[1] for feature in self.invariant_features])
        masked_invariant_mean[unnormalized_invariant_features] = 0
        masked_invariant_std[unnormalized_invariant_features] = 1

        unnormalized_covariant_features = ~np.array([feature[1] for feature in self.covariant_features])
        masked_covariant_mean[unnormalized_covariant_features] = 0
        masked_covariant_std[unnormalized_covariant_features] = 1

        return Statistics(
            torch.cat((masked_invariant_mean, masked_covariant_mean)),
            torch.cat((masked_invariant_std, masked_covariant_std))
        )

    def num_vectors(self) -> int:
        return self.invariant_mask.sum(1)

    def __getitem__(self, item) -> Source:
        invariant_data = self.invariant_data[item]
        covariant_data = self.covariant_data[item]

        invariant_data = invariant_data.unsqueeze(-3)

        invariant_data_shape = list(invariant_data.shape)
        invariant_data_shape[-3] = invariant_data_shape[-2]
        invariant_data = invariant_data.expand(invariant_data_shape)

        return Source(
            data=torch.cat((invariant_data, covariant_data), -1),
            mask=self.covariant_mask[item]
        )
