import torch
from torch import Tensor
from typing import Tuple

from spanet.dataset.regressions.base_regression import Regression


class GaussianRegression(Regression):
    @staticmethod
    def name():
        return "gaussian"

    @staticmethod
    def statistics(data: Tensor) -> Tuple[Tensor, Tensor]:
        mean = torch.nanmean(data)
        std = torch.sqrt(torch.nanmean(torch.square(data)) - torch.square(mean))

        return mean, std

    @staticmethod
    def loss(predictions: Tensor, targets: Tensor) -> Tensor:
        return torch.square(predictions - targets)
