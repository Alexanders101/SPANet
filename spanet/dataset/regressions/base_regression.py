from abc import ABC, abstractmethod
from typing import Tuple
from torch import Tensor


class Regression(ABC):
    @staticmethod
    @abstractmethod
    def name():
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def statistics(data: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def loss(predictions: Tensor, targets: Tensor) -> Tensor:
        raise NotImplementedError()

    @staticmethod
    def normalize(data: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        return (data - mean) / std

    @staticmethod
    def denormalize(data: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        return std * data + mean
