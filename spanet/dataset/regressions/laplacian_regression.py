import torch
from torch import Tensor
from typing import Tuple

from spanet.dataset.regressions.base_regression import Regression


class LaplacianRegression(Regression):
    @staticmethod
    def name():
        return "laplacian"

    @staticmethod
    def statistics(data: Tensor) -> Tuple[Tensor, Tensor]:
        valid_data = data[~torch.isnan(data)]

        median = torch.median(valid_data)
        deviation = torch.mean(torch.abs(valid_data - median))

        return median, deviation

    @staticmethod
    def loss(predictions: Tensor, targets: Tensor) -> Tensor:
        return torch.abs(predictions - targets)
