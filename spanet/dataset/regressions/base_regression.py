from abc import ABC, abstractmethod
from torch import Tensor

from spanet.dataset.types import Statistics

class Regression(ABC):
    @staticmethod
    @abstractmethod
    def name() -> str:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def statistics(data: Tensor) -> Statistics:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def loss(predictions: Tensor, targets: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        raise NotImplementedError()

    @staticmethod
    def normalize(data: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        return (data - mean) / std

    @staticmethod
    def denormalize(data: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        return std * data + mean
