import torch
from torch import Tensor

from spanet.dataset.regressions.base_regression import Regression, Statistics


class GaussianRegression(Regression):
    @staticmethod
    def name():
        return "gaussian"

    @staticmethod
    def statistics(data: Tensor) -> Statistics:
        mean = torch.nanmean(data)
        std = torch.sqrt(torch.nanmean(torch.square(data)) - torch.square(mean))

        return Statistics(mean, std)

    @staticmethod
    def loss(predictions: Tensor, targets: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        return torch.square((predictions - targets) / std)
