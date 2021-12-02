from typing import Union, Tuple, List, Optional, Mapping, Dict
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

# Special Keys used to describe event information.
SpecialKey = EventInfo.SpecialKey


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
            if ".ini" in event_info:
                self.event_info = EventInfo.read_from_ini(event_info)
            else:
                self.event_info = EventInfo.read_from_yaml(event_info)

        self.target_symmetries = self.event_info.mapped_targets.items()
        self.source_normalization = self.event_info.input_features
        self.event_transpositions = self.event_info.event_transpositions
        self.event_permutation_group = self.event_info.event_permutation_group
        self.unordered_event_transpositions = set(map(tuple, map(sorted, self.event_transpositions)))

        self.mean = None
        self.std = None

        with h5py.File(self.data_file, 'r') as file:
            # Get the first sequential input to find the total number of events in the dataset.
            first_sequential_group = [SpecialKey.Inputs, self.event_info.sequential_inputs[0]]
            self.num_events = self.dataset(file, first_sequential_group, SpecialKey.Mask).shape[0]

            limit_index = self.compute_limit_index(limit_index, randomization_seed)

            self.source_data = OrderedDict()
            self.source_mask = OrderedDict()

            for input_name in self.event_info.input_names:
                if self.event_info.input_type(input_name) == self.event_info.InputType.Sequential:
                    source_data, source_mask = self.load_sequential_data(file, input_name, limit_index)
                else:
                    source_data, source_mask = self.load_global_data(file, input_name, limit_index)

                self.source_data[input_name] = source_data
                self.source_mask[input_name] = source_mask

            self.targets = self.load_targets(file, limit_index)
            self.regressions = self.load_regressions(file, limit_index)

            self.num_events = limit_index.shape[0]

            self.num_sequential_vectors = sum(
                self.source_mask[input_name].sum(1)
                for input_name in self.event_info.sequential_inputs
            )

            self.num_global_vectors = sum(
                self.source_mask[input_name].sum(1)
                for input_name in self.event_info.global_inputs
            )

            self.num_vectors = self.num_sequential_vectors + self.num_global_vectors

            print(f"Index Range: {limit_index}")

            if not partial_events:
                self.limit_dataset_to_full_events()
                print(f"Training on Full Events only.")

            if jet_limit > 0:
                self.limit_dataset_to_jet_count(jet_limit)

    @staticmethod
    def dataset(hdf5_file: h5py.File, group: List[str], key: str) -> h5py.Dataset:
        group_string = "/".join(group)
        key_string = "/".join(group + [key])
        if key in hdf5_file[group_string]:
            return hdf5_file[key_string]
        else:
            raise KeyError(f"{key} not found in group {group_string}")

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

    def load_sequential_data(self, hdf5_file: h5py.File, input_name: str, limit_index: np.ndarray) -> Tuple[Tensor, Tensor]:
        """ Load source jet data and masking information

        Parameters
        ----------
        hdf5_file: h5py.File
            HDF5 file containing the event.
        input_name: str
            Which inputs to load from the file.
            SOURCE for old format, and the name of the input type for new format.
        limit_index: array or Tensor
            The limiting array for selecting a subset of dataset for this object.

        Returns
        -------
        torch.Tensor
            Source features

        torch.Tensor
            Source mask
        """
        input_group = [SpecialKey.Inputs, input_name]

        # Load in the mask for this vector input
        source_mask = torch.from_numpy(self.dataset(hdf5_file, input_group, SpecialKey.Mask)[:]).contiguous()

        # Load in vector features into a pre-made buffer
        num_jets = source_mask.shape[1]
        num_features = self.event_info.num_features(input_name)
        source_data = torch.empty(num_features, self.num_events, num_jets, dtype=torch.float32)

        for index, (feature, _, log_transform) in enumerate(self.event_info.input_features[input_name]):
            self.dataset(hdf5_file, input_group, feature).read_direct(source_data[index].numpy())
            if log_transform:
                source_data[index] = torch.log(torch.clamp(source_data[index], min=1e-6)) * source_mask

        # Reshape and limit data to the limiting index.
        source_data = source_data.permute(1, 2, 0)
        source_data = source_data[limit_index].contiguous()
        source_mask = source_mask[limit_index].contiguous()

        return source_data, source_mask

    def load_global_data(self, hdf5_file: h5py.File, input_name: str, limit_index: np.ndarray) -> Tuple[Tensor, Tensor]:
        """ Load source jet data and masking information

        Parameters
        ----------
        hdf5_file: h5py.File
            HDF5 file containing the event.
        input_name: str
            Which inputs to load from the file.
            SOURCE for old format, and the name of the input type for new format.
        limit_index: array or Tensor
            The limiting array for selecting a subset of dataset for this object.

        Returns
        -------
        torch.Tensor
            Source features

        torch.Tensor
            Source mask
        """
        input_group = [SpecialKey.Inputs, input_name]

        # Try and load a mask for this global features. If none is present, assume all vectors are valid.
        try:
            source_mask = torch.from_numpy(self.dataset(hdf5_file, input_group, SpecialKey.Mask)[:]).contiguous()
        except KeyError:
            source_mask = torch.ones(self.num_events, dtype=torch.bool)

        # Load in vector features.
        num_features = self.event_info.num_features(input_name)
        source_data = torch.empty(num_features, self.num_events, dtype=torch.float32)

        for index, (feature, _, log_transform) in enumerate(self.event_info.input_features[input_name]):
            self.dataset(hdf5_file, input_group, feature).read_direct(source_data[index].numpy())
            if log_transform:
                source_data[index] = torch.log(torch.clamp(source_data[index], min=1e-6)) * source_mask

        # Reshape and limit data to the limiting index.
        source_data = source_data.transpose(0, 1)
        source_data = source_data[limit_index].contiguous()
        source_mask = source_mask[limit_index].contiguous()

        # Add a fake timestep dimension to global vectors.
        source_data = source_data.unsqueeze(1)
        source_mask = source_mask.unsqueeze(1)

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
            target_data = torch.empty(len(jets), self.num_events, dtype=torch.int64)

            for index, jet in enumerate(jets):
                self.dataset(hdf5_file, [SpecialKey.Targets, target], jet).read_direct(target_data[index].numpy())

            target_data = target_data.transpose(0, 1)
            target_data = target_data[limit_index]

            target_mask = (target_data >= 0).all(1)

            targets[target] = (target_data, target_mask)

        return targets

    def load_regressions(self, hdf5_file: h5py.File, limit_index: np.ndarray) -> Mapping[str, Tuple[Tensor, Tensor]]:
        """ Load regression target data

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
        def load_regression(group):
            regression_info = self.event_info.regressions

            for key in group:
                regression_info = regression_info[key]

            if len(regression_info) == 0:
                return None

            regression_data = torch.empty(len(regression_info), self.num_events, dtype=torch.float32)

            for index, key in enumerate(regression_info):
                current_dataset = self.dataset(hdf5_file, [SpecialKey.Regressions, *group], key)
                current_dataset.read_direct(regression_data[index].numpy())

            regression_data = regression_data.transpose(0, 1)
            regression_data = regression_data[limit_index]

            return regression_data

        regressions = OrderedDict()
        regressions[SpecialKey.Event] = load_regression([SpecialKey.Event])
        for particle in self.event_info.targets:
            regressions["/".join((particle, SpecialKey.Particle))] = load_regression([particle, SpecialKey.Particle])

            for daughter in self.event_info.targets[particle][0]:
                regressions["/".join((particle, daughter))] = load_regression([particle, daughter])

        return regressions

    def compute_statistics(self,
                           mean: Optional[Mapping[str, Tensor]] = None,
                           std: Optional[Mapping[str, Tensor]] = None) -> Tuple[Mapping[str, Tensor], Mapping[str, Tensor]]:
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
            mean = OrderedDict()
            std = OrderedDict()

            for input_name, source_data in self.source_data.items():
                masked_data = source_data[self.source_mask[input_name]]
                masked_mean = masked_data.mean(0)
                masked_std = masked_data.std(0)

                masked_std[masked_std < 1e-5] = 1

                masked_mean[~self.event_info.normalized_features(input_name)] = 0
                masked_std[~self.event_info.normalized_features(input_name)] = 1

                mean[input_name] = masked_mean
                std[input_name] = masked_std

        self.mean = mean
        self.std = std

        return mean, std

    def compute_regression_statistics(self) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """ Compute the target regression statistics

        Returns
        -------
        (Dict[str, Tensor], Dict[str, Tensor])
            The mean and standard deviation for existing regression values.
        """

        regression_means = {
            key: value.mean(0, keepdim=True)
            for key, value in self.regressions.items()
            if value is not None
        }

        regression_stds = {
            key: value.std(0, keepdim=True)
            for key, value in self.regressions.items()
            if value is not None
        }

        return regression_means, regression_stds

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
        max_jets = self.num_sequential_vectors.max()
        min_jets = self.num_sequential_vectors.min()

        class_count = torch.bincount(self.num_sequential_vectors, minlength=max_jets + 1)

        # Compute the effective class count
        # https://arxiv.org/pdf/1901.05555.pdf
        beta = 1 - (1 / self.num_sequential_vectors.shape[0])
        jet_class_weights = (1 - beta) / (1 - (beta ** class_count))
        jet_class_weights[torch.isinf(jet_class_weights)] = 0
        jet_class_weights = (max_jets - min_jets + 1) * jet_class_weights / jet_class_weights.sum()

        return jet_class_weights

    def limit_dataset_to_mask(self, event_mask):
        for input_name, source_data in self.source_data.items():
            source_mask = self.source_mask[input_name]

            self.source_data[input_name] = source_data[event_mask].contiguous()
            self.source_mask[input_name] = source_mask[event_mask].contiguous()

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
        self.limit_dataset_to_mask(self.num_sequential_vectors == jet_count)

    def __len__(self) -> int:
        return self.num_events

    def __getitem__(self, item) -> Tuple[
        Tuple[Tuple[Tensor, Tensor], ...],
        Tensor,
        Tuple[Tuple[Tensor, Tensor], ...],
        Dict[str, Tensor]
    ]:
        sources = tuple(
            (self.source_data[input_name][item], self.source_mask[input_name][item])
            for input_name in self.source_data
        )

        targets = tuple(
            (target_data[item], target_mask[item])
            for target_data, target_mask in self.targets.values()
        )

        regressions = {
            key: value[item]
            for key, value in self.regressions.items()
            if value is not None
        }

        return sources, self.num_sequential_vectors[item], targets, regressions
