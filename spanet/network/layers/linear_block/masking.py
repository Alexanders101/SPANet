import torch
from torch import nn


# noinspection PyMethodMayBeStatic
class IdentityMasking(nn.Module):
    def forward(self, values, sequence_mask):
        return values


# noinspection PyMethodMayBeStatic
class MultiplicativeMasking(nn.Module):
    def forward(self, values, sequence_mask):
        return values * sequence_mask.to(values.dtype)


# noinspection PyMethodMayBeStatic
class FillingMasking(nn.Module):
    def forward(self, values, sequence_mask):
        return torch.masked_fill(values, ~sequence_mask, 0.0)


# noinspection SpellCheckingInspection
def create_masking(masking: str) -> nn.Module:
    masking = masking.lower().replace("_", "").replace(" ", "")

    if masking == "multiplicative":
        return MultiplicativeMasking()
    elif masking == "filling":
        return FillingMasking()
    else:
        return IdentityMasking()
