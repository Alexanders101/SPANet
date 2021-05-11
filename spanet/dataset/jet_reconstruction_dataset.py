from typing import Union, Tuple, List, Optional, Mapping
from collections import OrderedDict

import numpy as np
import torch
import h5py

# noinspection PyProtectedMember
from torch.utils.data import Dataset
from torch import Tensor

from spanet.dataset.event_info import EventInfo

# The possible types for the limit index parameter.
TLimitIndex = Union[Tuple[float, float], List[float], float, np.ndarray, Tensor]


class JetReconstructionDataset(Dataset):
    def __init__(self,
                 data_file: str,
                 event_info: Union[str, EventInfo],
                 limit_index: TLimitIndex = 1.0,
                 randomization_seed: int = 0,
                 jet_limit: int = 0,
                 partial_events: bool = True):
        """ A container class for reading in jet reconstruction datasets.

        Parameters
        ----------
        data_file : str
            HDF5 file containing the jet event data, see Notes section for structure information.
        event_info : str or EventInfo
            An EventInfo object which contains the symmetries for the event. Or the path of the ini file where the
            event info is defined. See feynman.dataset.EventInfo.
        limit_index : float in [-1, 1], tuple of floats, or array-like.
            If a positive float - limit the dataset to the first limit_index percent of the data.
            If a negative float - limit the dataset to the last |limit_index| percent of the data.
            If a tuple - limit the dataset to [limit_index[0], limit_index[1]] percent of the data.
            If array-like or tensor - limit the dataset to the specified indices.
        partial_events : bool
            Whether or not to allow training on partial events as well as complete events.

        Notes
        -----
        Data file structure:
        source/FEATURE_NAME - (num_events, num_jets, ) - The padded jet data for the event.
        source/mask - (num_events, num_jets) - boolean value if a given jet is in the event or a padding jet.

        For each target in your event:
        TARGET_NAME/JET_NAME - (num_events, ) - Indices indicating which of the source jets are the labels sub-jets.
        TARGET_NAME/mask - (num_events, ) - Boolean value if the given target is present in the event at all.

        TARGET_NAMES must match the targets defined in the event_info.
        """
        super(JetReconstructionDataset, self).__init__()

        self.data_file = data_file
        self.event_info: EventInfo = event_info
        if isinstance(event_info, str):
            self.event_info = EventInfo.read_from_ini(event_info)

        self.target_symmetries = self.event_info.mapped_targets.items()
        self.source_normalization = self.event_info.source_features
        self.event_transpositions = self.event_info.event_transpositions
        self.event_permutation_group = self.event_info.event_permutation_group
        self.unordered_event_transpositions = set(map(tuple, map(sorted, self.event_transpositions)))

        self.mean = None
        self.std = None

        with h5py.File(self.data_file, 'r') as file:
            self.num_events, self.num_jets = file["source/mask"].shape
            self.num_features = self.event_info.num_features

            limit_index = self.compute_limit_index(limit_index, randomization_seed)

            self.source_data, self.source_mask = self.load_source_data(file, limit_index)
            self.targets = self.load_targets(file, limit_index)

            self.num_events = limit_index.shape[0]
            print(f"Index Range: {limit_index}")

            if not partial_events:
                self.limit_dataset_to_full_events()
                print(f"Full Events only.")

            if jet_limit > 0:
                self.limit_dataset_to_jet_count(jet_limit)

    def compute_limit_index(self, limit_index: TLimitIndex, randomization_seed: int) -> np.ndarray:
        """ Take subsection of the data for training / validation

        Parameters
        ----------
        limit_index : float in [-1, 1], tuple of floats, or array-like
            If a positive float - limit the dataset to the first limit_index percent of the data
            If a negative float - limit the dataset to the last |limit_index| percent of the data
            If a tuple - limit the dataset to [limit_index[0], limit_index[1]] percent of the data
            If array-like or tensor - limit the dataset to the specified indices.
        randomization_seed: int
            If randomization_seed is non-zero, then we will first shuffle the indices in a deterministic manner
            before taking the subset defined by `limit_index`.

        Returns
        -------
        np.ndarray or torch.Tensor
        """
        # In the float case, we just generate the list with the appropriate bounds
        if isinstance(limit_index, float):
            limit_index = (0.0, limit_index) if limit_index > 0 else (1.0 + limit_index, 1.0)

        # In the list / tuple case, we want a contiguous range
        if isinstance(limit_index, (list, tuple)):
            lower_index = int(round(limit_index[0] * self.num_events))
            upper_index = int(round(limit_index[1] * self.num_events))

            if randomization_seed > 0:
                random_state = np.random.RandomState(seed=randomization_seed)
                limit_index = random_state.permutation(self.num_events)
            else:
                limit_index = np.arange(self.num_events)

            limit_index = limit_index[lower_index:upper_index]

        # Convert to numpy array for simplicity
        if isinstance(limit_index, Tensor):
            limit_index = limit_index.numpy()

        # Make sure the resulting index array is sorted for faster loading.
        return np.sort(limit_index)

    def load_source_data(self, hdf5_file: h5py.File, limit_index: np.ndarray) -> Tuple[Tensor, Tensor]:
        """ Load source jet data and masking information

        Parameters
        ----------
        hdf5_file: h5py.File
            HDF5 file containing the event.
        limit_index: array or Tensor
            The limiting array for selecting a subset of dataset for this object.

        Returns
        -------
        torch.Tensor
            Source features

        torch.Tensor
            Source mask
        """
        source_mask = torch.from_numpy(hdf5_file["source/mask"][:]).contiguous()
        source_data = torch.empty(self.num_features, self.num_events, self.num_jets, dtype=torch.float32)

        for index, (feature, _, log_transform) in enumerate(self.event_info.source_features):
            hdf5_file[f"source/{feature}"].read_direct(source_data[index].numpy())
            if log_transform:
                source_data[index] = torch.log(torch.clamp(source_data[index], min=1e-6)) * source_mask

        source_data = source_data.permute(1, 2, 0)
        source_data = source_data[limit_index].contiguous()
        source_mask = source_mask[limit_index].contiguous()

        return source_data, source_mask

    def load_targets(self, hdf5_file: h5py.File, limit_index: np.ndarray) -> Mapping[str, Tuple[Tensor, Tensor]]:
        """ Load target indices for every defined target

        Parameters
        ----------
        hdf5_file: h5py.File
            HDF5 file containing the event.
        limit_index: array or Tensor
            The limiting array for selecting a subset of dataset for this object.

        Returns
        -------
        OrderedDict: str -> (Tensor, Tensor)
            A dictionary mapping the target name to the target indices and mask.
        """
        targets = OrderedDict()
        for target, (jets, _) in self.event_info.targets.items():
            target_mask = torch.from_numpy(hdf5_file[f"{target}/mask"][:][limit_index])
            target_data = torch.empty(len(jets), self.num_events, dtype=torch.int64)

            for index, jet in enumerate(jets):
                hdf5_file[f"{target}/{jet}"].read_direct(target_data[index].numpy())

            target_data = target_data.transpose(0, 1)
            target_data = target_data[limit_index]

            targets[target] = (target_data, target_mask)

        return targets

    def compute_statistics(self, mean: Optional[Tensor] = None, std: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """ Compute the mean and standard deviation of features with normalization enabled in the event file.

        Parameters
        ----------
        mean: Tensor, optional
        std: Tensor, optional
            Give existing values for mean and standard deviation to set this value
            dataset's statistics to those values. This is especially useful for
            normalizing the validation and testing datasets with training statistics.

        Returns
        -------
        (Tensor, Tensor)
            The new mean and standard deviation for this dataset.
        """
        if mean is None:
            masked_data = self.source_data[self.source_mask]
            mean = masked_data.mean(0)
            std = masked_data.std(0)

            std[std < 1e-5] = 1
            mean[~self.event_info.normalized_features] = 0
            std[~self.event_info.normalized_features] = 1

        self.mean = mean
        self.std = std

        return mean, std

    def compute_particle_balance(self):
        # Extract just the mask information from the dataset.
        masks = torch.stack([target[1] for target in self.targets.values()])

        eq_class_counts = {}
        num_targets = masks.shape[0]
        full_targets = frozenset(range(num_targets))

        # Find the count for every equivalence class in the masks.
        for eq_class in self.event_info.event_equivalence_classes:
            eq_class_count = 0

            for positive_target in eq_class:
                negative_target = full_targets - positive_target

                # Note that we must ensure that every sample is assigned exactly one equivalence class.
                # So we have to make sure that the ONLY target present in the one we want.
                positive_target = masks[list(positive_target), :].all(0)
                negative_target = masks[list(negative_target), :].any(0)

                targets = positive_target & ~negative_target
                eq_class_count += targets.sum().item()

            eq_class_counts[eq_class] = eq_class_count

        # Compute the effective class count
        # https://arxiv.org/pdf/1901.05555.pdf
        beta = 1 - (10 ** -np.log10(masks.shape[1]))
        eq_class_weights = {key: (1 - beta) / (1 - (beta ** value)) for key, value in eq_class_counts.items()}
        target_weights = {target: weight for eq_class, weight in eq_class_weights.items() for target in eq_class}

        # Convert these target weights into a bit-mask indexed tensor
        norm = sum(eq_class_weights.values())
        index_tensor = 2 ** np.arange(num_targets)
        target_weights_tensor = torch.zeros(2 ** num_targets)

        for target, weight in target_weights.items():
            index = index_tensor[list(target)].sum()
            target_weights_tensor[index] = len(eq_class_weights) * weight / norm

        return torch.from_numpy(index_tensor), target_weights_tensor

    def compute_jet_balance(self):
        num_jets = self.source_mask.sum(1)
        max_jets = num_jets.max()
        min_jets = num_jets.min()

        class_count = torch.bincount(num_jets, minlength=max_jets + 1)

        # Compute the effective class count
        # https://arxiv.org/pdf/1901.05555.pdf
        beta = 1 - (1 / num_jets.shape[0])
        jet_class_weights = (1 - beta) / (1 - (beta ** class_count))
        jet_class_weights[torch.isinf(jet_class_weights)] = 0
        jet_class_weights = (max_jets - min_jets + 1) * jet_class_weights / jet_class_weights.sum()

        return jet_class_weights

    def limit_dataset_to_mask(self, event_mask):
        self.source_data = self.source_data[event_mask].contiguous()
        self.source_mask = self.source_mask[event_mask].contiguous()

        for key in self.targets:
            targets, masks = self.targets[key]

            targets = targets[event_mask].contiguous()
            masks = masks[event_mask].contiguous()

            self.targets[key] = (targets, masks)

        self.num_events = event_mask.sum().item()

    def limit_dataset_to_partial_events(self):
        target_masks = torch.stack([target[1] for target in self.targets.values()])
        non_empty_events = target_masks.any(0)
        self.limit_dataset_to_mask(non_empty_events)

    def limit_dataset_to_full_events(self):
        target_masks = torch.stack([target[1] for target in self.targets.values()])
        full_events = target_masks.all(0)
        self.limit_dataset_to_mask(full_events)

    def limit_dataset_to_jet_count(self, jet_count):
        event_mask = self.source_mask.sum(1) == jet_count
        self.limit_dataset_to_mask(event_mask)

    def limit_dataset_to_btag_count(self, btag_count):
        event_mask = self.source_data[:, :, -1].sum(1) < (btag_count - 0.5)
        self.limit_dataset_to_mask(event_mask)

    def __len__(self) -> int:
        return self.num_events

    def __getitem__(self, item) -> Tuple[Tuple[Tensor, Tensor], ...]:
        source = (self.source_data[item], self.source_mask[item])
        targets = ((target_data[item], target_mask[item]) for target_data, target_mask in self.targets.values())

        return source, *targets
