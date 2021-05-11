from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import init

# Pytorch Masked BatchNorm
# Based on this implementation
# https://gist.github.com/yangkky/364413426ec798589463a3a88be24219


class MaskedBatchNorm1D(nn.Module):

    __constants__ = ["num_features", "eps", "momentum", "affine", "track_running_stats"]

    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True):
        super(MaskedBatchNorm1D, self).__init__()

        self.track_running_stats = track_running_stats
        self.num_features = num_features
        self.momentum = momentum
        self.affine = affine
        self.eps = eps

        # Register affine transform learnable parameters
        if affine:
            self.weight = nn.Parameter(torch.Tensor(1, 1, num_features))
            self.bias = nn.Parameter(torch.Tensor(1, 1, num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        # Register moving average storable parameters
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(1, 1, num_features))
            self.register_buffer('running_var', torch.ones(1, 1, num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, features: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Calculate the masked mean and variance
        batch_size, timesteps, feature_dim = features.shape

        if mask is not None and mask.shape != (batch_size, timesteps):
            raise ValueError('Mask should have shape (B, ).')

        if feature_dim != self.num_features:
            raise ValueError('Expected %d channels but images has %d channels' % (self.num_features, feature_dim))

        if mask is not None:
            masked_features = features * mask.view(batch_size, timesteps, 1)
            normalization_factor = mask.sum()
        else:
            masked_features = features
            normalization_factor = torch.scalar_tensor(batch_size * timesteps, dtype=torch.int64)

        # Find the masked sum of the images
        masked_sum = masked_features.sum(dim=(0, 1), keepdim=True)

        # Compute masked image statistics
        current_mean = masked_sum / normalization_factor
        current_var = ((masked_features - current_mean) ** 2).sum(dim=(0, 1), keepdim=True) / normalization_factor

        # Update running statistics
        if self.track_running_stats and self.training:
            if self.num_batches_tracked == 0:
                self.running_mean = current_mean
                self.running_var = current_var
            else:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * current_mean.detach()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * current_var.detach()
            self.num_batches_tracked += 1

        # Apply running statistics transform
        if self.track_running_stats and not self.training:
            normed_images = (masked_features - self.running_mean) / (torch.sqrt(self.running_var + self.eps))
        else:
            normed_images = (masked_features - current_mean) / (torch.sqrt(current_var + self.eps))

        # Apply affine transform from learned parameters
        if self.affine:
            normed_images = normed_images * self.weight + self.bias

        return normed_images
