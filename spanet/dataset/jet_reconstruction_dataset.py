import functools
from typing import Union, Tuple, List, Optional, Dict
from collections import OrderedDict

import h5py
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset

from spanet.dataset.types import SpecialKey, NDArray, Batch, AssignmentTargets, Source, ArrayLike
from spanet.dataset.event_info import EventInfo
from spanet.dataset.inputs import create_source_input
from spanet.dataset.types import InputType
from spanet.dataset.regressions import regression_statistics

# The possible types for the limit index parameter.
TLimitIndex = Union[
    Tuple[float, float],
    List[float],
    float,
    np.ndarray,
    Tensor
]

# The format of a batch produced by this dataset
TBatch = Tuple[
    Tuple[Tuple[Tensor, Tensor], ...],
    Tensor,
    Tuple[Tuple[Tensor, Tensor], ...],
    Dict[str, Tensor],
    Dict[str, Tensor]
]


class JetReconstructionDataset(Dataset):
    def __init__(
        self,
        data_file: str,
        event_info: Union[str, EventInfo],
        limit_index: TLimitIndex = 1.0,
        randomization_seed: int = 0,
        vector_limit: int = 0,
        partial_events: bool = True
    ):
        """ A container class for reading in jet reconstruction datasets.

        Parameters
        ----------
        data_file : str
            HDF5 file containing the jet event data, see Notes section for structure information.
        event_info : str or EventInfo
            An EventInfo object which contains the symmetries for the event.
            Or the path of the yaml file where the event info is defined.
            See `feynman.dataset.EventInfo`
        limit_index : float in [-1, 1], tuple of floats, or array-like.
            If a positive float - limit the dataset to the first limit_index percent of the data.
            If a negative float - limit the dataset to the last |limit_index| percent of the data.
            If a tuple - limit the dataset to [limit_index[0], limit_index[1]] percent of the data.
            If array-like or tensor - limit the dataset to the specified indices.
        randomization_seed: int
            If set to a value greater than 0, randomize the order of the dataset. Applied before limit index.
        vector_limit: int
            Limit the event to a specific number of vectors.
        partial_events : bool
            Whether to allow training on partial events, not just complete events.
        """
        super(JetReconstructionDataset, self).__init__()

        self.data_file = data_file
        self.event_info: EventInfo = event_info

        if isinstance(event_info, str):
            if ".ini" in event_info:
                self.event_info = EventInfo.read_from_ini(event_info)
            else:
                self.event_info = EventInfo.read_from_yaml(event_info)

        self.mean = None
        self.std = None

        with h5py.File(self.data_file, 'r') as file:
            # Get the first merged_momenta input to find the total number of events in the dataset.
            first_key = [
                name
                for name, input_type in self.event_info.input_types.items()
                if input_type == InputType.Sequential
            ][0]
            self.num_events = self.dataset(file, [SpecialKey.Inputs, first_key], SpecialKey.Mask).shape[0]

            # Adjust limit index into a standard format.
            limit_index = self.compute_limit_index(limit_index, randomization_seed)

            # Load source features from hdf5 file, processing them depending on their type.
            self.sources = OrderedDict((
                (input_name, create_source_input(self.event_info, file, input_name, self.num_events, limit_index))
                for input_name in self.event_info.input_names
            ))

            # Load various types of targets.
            self.assignments = self.load_assignments(file, limit_index)
            self.regressions, self.regression_types = self.load_regressions(file, limit_index)
            self.classifications = self.load_classifications(file, limit_index)

            # Update size information after loading and limiting dataset.
            self.num_events = limit_index.shape[0]
            self.num_vectors = sum(source.num_vectors() for source in self.sources.values())

            print(f"Index Range: {limit_index[0]}...{limit_index[-1]}")

        # Optionally remove any events where any of the targets are missing.
        if not partial_events:
            self.limit_dataset_to_full_events()
            print(f"Training on Full Events only.")

        # Optionally limit the dataset to a specific number of jets.
        if vector_limit > 0:
            self.limit_dataset_to_jet_count(vector_limit)

    @staticmethod
    def dataset(hdf5_file: h5py.File, group: List[str], key: str) -> h5py.Dataset:
        group_string = "/".join(group)
        key_string = "/".join(group + [key])
        if key in hdf5_file[group_string]:
            return hdf5_file[key_string]
        else:
            raise KeyError(f"{key} not found in group {group_string}")

    def compute_limit_index(self, limit_index: TLimitIndex, randomization_seed: int) -> NDArray[np.int64]:
        """ Take subsection of the data for training / validation

        Parameters
        ----------
        limit_index : float in [-1, 1], tuple of floats, or array-like
            If a positive float - limit the dataset to the FIRST limit_index percent of the data
            If a negative float - limit the dataset to the LAST |limit_index| percent of the data
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

    def load_assignments(self, hdf5_file: h5py.File, limit_index: np.ndarray) -> Dict[str, Tuple[Tensor, Tensor]]:
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
        for event_particle, daughter_particles in self.event_info.product_particles.items():
            target_data = torch.empty(len(daughter_particles), self.num_events, dtype=torch.int64)

            for index, daughter in enumerate(daughter_particles):
                dataset = self.dataset(hdf5_file, [SpecialKey.Targets, event_particle], daughter)
                dataset.read_direct(target_data[index].numpy())

            target_data = target_data.transpose(0, 1)
            target_data = target_data[limit_index]

            target_mask = (target_data >= 0).all(1)

            targets[event_particle] = (target_data, target_mask)

        return targets

    def tree_key_data(self, hdf5_file: h5py.File, limit_index, root, group, index):
        key = "/".join((*group, index))
        data = self.dataset(hdf5_file, [root, *group], index)
        data = torch.from_numpy(data[:][limit_index])
        return key, data

    def load_regressions(self, hdf5_file: h5py.File, limit_index: np.ndarray) -> Tuple[Dict[str, Tensor], Dict[str, str]]:
        tree_key_data = functools.partial(self.tree_key_data, hdf5_file, limit_index, SpecialKey.Regressions)
        targets = OrderedDict()
        types = OrderedDict()

        for target in self.event_info.regressions[SpecialKey.Event]:
            key, data = tree_key_data([SpecialKey.Event], target.name)
            targets[key] = data
            types[key] = target.type

        for particle in self.event_info.event_particles:
            for target in self.event_info.regressions[particle][SpecialKey.Particle]:
                key, data = tree_key_data([particle, SpecialKey.Particle], target.name)
                targets[key] = data
                types[key] = target.type

            for daughter in self.event_info.product_particles[particle]:
                for target in self.event_info.regressions[particle][daughter]:
                    key, data = tree_key_data([particle, daughter], target.name)
                    targets[key] = data
                    types[key] = target.type

        return targets, types

    def load_classifications(self, hdf5_file: h5py.File, limit_index: np.ndarray) -> Dict[str, Tensor]:
        tree_key_data = functools.partial(self.tree_key_data, hdf5_file, limit_index, SpecialKey.Classifications)

        targets = OrderedDict()

        def add_target(key, value):
            targets[key] = value

        for target in self.event_info.classifications[SpecialKey.Event]:
            add_target(*tree_key_data([SpecialKey.Event], target))

        for particle in self.event_info.product_particles:
            for target in self.event_info.classifications[particle][SpecialKey.Particle]:
                add_target(*tree_key_data([particle, SpecialKey.Particle], target))

            for daughter in self.event_info.product_particles[particle]:
                for target in self.event_info.classifications[particle][daughter]:
                    add_target(*tree_key_data([particle, daughter], target))

        return targets

    def compute_source_statistics(
            self,
            mean: Optional[Dict[str, Tensor]] = None,
            std: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
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

            for input_name, source in self.sources.items():
                mean[input_name], std[input_name] = source.compute_statistics()

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
        regression_means = OrderedDict()
        regression_stds = OrderedDict()

        for key, value in self.regressions.items():
            if value is None:
                continue

            mean, std = regression_statistics(self.regression_types[key])(value)
            regression_means[key] = mean
            regression_stds[key] = std

        return regression_means, regression_stds

    def compute_classification_class_counts(self) -> Dict[str, int]:
        return OrderedDict((
            (key, value.max().item() + 1)
            for key, value in self.classifications.items()
            if value is not None
        ))

    def compute_particle_balance(self):
        # Extract just the mask information from the dataset.
        masks = torch.stack([target[1] for target in self.assignments.values()])

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

    def compute_vector_balance(self):
        max_vectors = self.num_vectors.max()
        min_vectors = self.num_vectors.min()

        class_count = torch.bincount(self.num_vectors, minlength=max_vectors + 1)

        # Compute the effective class count
        # https://arxiv.org/pdf/1901.05555.pdf
        beta = 1 - (1 / self.num_vectors.shape[0])
        vector_class_weights = (1 - beta) / (1 - (beta ** class_count))
        vector_class_weights[torch.isinf(vector_class_weights)] = 0
        vector_class_weights = (max_vectors - min_vectors + 1) * vector_class_weights / vector_class_weights.sum()

        return vector_class_weights

    def compute_classification_balance(self):
        def compute_effective_counts(targets):
            beta = 1 - (1 / targets.shape[0])
            vector_class_weights = (1 - beta) / (1 - (beta ** torch.bincount(targets)))
            vector_class_weights[torch.isinf(vector_class_weights)] = 0
            vector_class_weights = vector_class_weights.shape[0] * vector_class_weights / vector_class_weights.sum()

            return vector_class_weights

        return OrderedDict((
            (key, compute_effective_counts(value))
            for key, value in self.classifications.items()
            if value is not None
        ))

    def limit_dataset_to_mask(self, event_mask: Tensor):
        for input_name, source in self.sources.items():
            source.limit(event_mask)

        for key in self.assignments:
            assignments, masks = self.assignments[key]

            assignments = assignments[event_mask].contiguous()
            masks = masks[event_mask].contiguous()

            self.assignments[key] = (assignments, masks)

        for key, regressions in self.regressions.items():
            self.regressions[key] = regressions[event_mask]

        for key, classifications in self.classifications.items():
            self.classifications[key] = classifications[event_mask]

        self.num_events = event_mask.sum().item()
        self.num_vectors = sum(source.num_vectors() for source in self.sources.values())

    def limit_dataset_to_partial_events(self):
        vector_masks = torch.stack([target[1] for target in self.assignments.values()])
        non_empty_events = vector_masks.any(0)
        self.limit_dataset_to_mask(non_empty_events)

    def limit_dataset_to_full_events(self):
        vector_masks = torch.stack([target[1] for target in self.assignments.values()])
        full_events = vector_masks.all(0)
        self.limit_dataset_to_mask(full_events)

    def limit_dataset_to_jet_count(self, jet_count):
        self.limit_dataset_to_mask(self.num_vectors == jet_count)

    def __len__(self) -> int:
        return self.num_events

    def __getitem__(self, item) -> Batch:
        sources = tuple(
            source[item]
            for source in self.sources.values()
        )

        assignments = tuple(
            AssignmentTargets(assignment[item], mask[item])
            for assignment, mask in self.assignments.values()
        )

        regressions = {
            key: value[item]
            for key, value in self.regressions.items()
            if value is not None
        }

        classifications = {
            key: value[item]
            for key, value in self.classifications.items()
            if value is not None
        }

        return Batch(
            sources,
            self.num_vectors[item],
            assignments,
            regressions,
            classifications
        )
