from abc import ABC, abstractmethod
from typing import List
import h5py

import numpy as np

from spanet.dataset.event_info import EventInfo
from spanet.dataset.types import Statistics, Source


class BaseInput(ABC):
    def __init__(
            self,
            event_info: EventInfo,
            hdf5_file: h5py.File,
            input_name: str,
            num_events: int,
            limit_index: np.ndarray
    ):
        super(BaseInput, self).__init__()

        self.input_name = input_name
        self.event_info = event_info
        self.num_events = num_events
        self.input_features = self.event_info.input_features[input_name]

        self.load(hdf5_file, limit_index)

    @staticmethod
    def dataset(hdf5_file: h5py.File, group: List[str], key: str) -> h5py.Dataset:
        group_string = "/".join(group)
        key_string = "/".join(group + [key])
        if key in hdf5_file[group_string]:
            return hdf5_file[key_string]
        else:
            raise KeyError(f"{key} not found in group {group_string}")

    @abstractmethod
    def load(self, hdf5_file: h5py.File, limit_index: np.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def limit(self, event_mask):
        raise NotImplementedError()

    @abstractmethod
    def compute_statistics(self) -> Statistics:
        raise NotImplementedError()

    @abstractmethod
    def num_vectors(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, item) -> Source:
        raise NotImplementedError()
