import h5py
import numpy as np

import torch

from spanet.dataset.types import SpecialKey, Statistics, Source
from spanet.dataset.inputs.BaseInput import BaseInput



class SequentialInput(BaseInput):

    # noinspection PyAttributeOutsideInit
    def load(self, hdf5_file: h5py.File, limit_index: np.ndarray):
        input_group = [SpecialKey.Inputs, self.input_name]

        # Load in the mask for this vector input
        source_mask = torch.from_numpy(self.dataset(hdf5_file, input_group, SpecialKey.Mask)[:]).contiguous()

        # Load in vector features into a pre-made buffer
        num_jets = source_mask.shape[1]
        num_features = self.event_info.num_features(self.input_name)
        source_data = torch.empty(num_features, self.num_events, num_jets, dtype=torch.float32)

        for index, (feature, _, log_transform) in enumerate(self.event_info.input_features[self.input_name]):
            self.dataset(hdf5_file, input_group, feature).read_direct(source_data[index].numpy())
            if log_transform:
                # torch.clamp_(source_data[index], min=1e-6)
                source_data[index] += 1
                torch.log_(source_data[index])
                source_data[index] *= source_mask

        # Reshape and limit data to the limiting index.
        source_data = source_data.permute(1, 2, 0)
        self.source_data = source_data[limit_index].contiguous()
        self.source_mask = source_mask[limit_index].contiguous()

    # noinspection PyAttributeOutsideInit
    def limit(self, event_mask):
        self.source_data = self.source_data[event_mask].contiguous()
        self.source_mask = self.source_mask[event_mask].contiguous()

    def compute_statistics(self) -> Statistics:
        masked_data = self.source_data[self.source_mask]
        masked_mean = masked_data.mean(0)
        masked_std = masked_data.std(0)

        masked_std[masked_std < 1e-5] = 1

        masked_mean[~self.event_info.normalized_features(self.input_name)] = 0
        masked_std[~self.event_info.normalized_features(self.input_name)] = 1

        return Statistics(masked_mean, masked_std)

    def num_vectors(self) -> int:
        return self.source_mask.sum(1)

    def __getitem__(self, item) -> Source:
        return Source(self.source_data[item], self.source_mask[item])
