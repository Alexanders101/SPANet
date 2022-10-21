import torch
from torch import Tensor

from spanet.dataset.regressions.base_regression import Regression, Statistics


class LaplacianRegression(Regression):
    @staticmethod
    def name():
        return "laplacian"

    @staticmethod
    def statistics(data: Tensor) -> Statistics:
        valid_data = data[~torch.isnan(data)]

        median = torch.median(valid_data)
        deviation = torch.mean(torch.abs(valid_data - median))

        return Statistics(median, deviation)

    @staticmethod
    def loss(predictions: Tensor, targets: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        return torch.abs(predictions - targets) / std
