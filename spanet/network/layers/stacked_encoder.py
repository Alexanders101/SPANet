from typing import Tuple, Optional

from torch import Tensor, nn, jit

from spanet.options import Options
from spanet.network.layers.linear_block import LinearStack


# noinspection PyUnusedLocal, PyMethodMayBeStatic
class LinearIdentity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(LinearIdentity, self).__init__()

    def forward(self, x: Tensor, sequence_mask: Tensor) -> Tensor:
        return x


# noinspection PyUnusedLocal, PyMethodMayBeStatic
class TransformerIdentity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TransformerIdentity, self).__init__()

    def forward(self, x: Tensor, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        return x


class StackedEncoder(nn.Module):
    def __init__(self,
                 options: Options,
                 num_embedding_layers: int,
                 num_encoder_layers: int,
                 transformer_options: Tuple[int, int, int, float, str]):

        super(StackedEncoder, self).__init__()

        if num_embedding_layers > 0:
            self.embedding = LinearStack(options, num_embedding_layers, options.hidden_dim, options.skip_connections)
        else:
            self.embedding = LinearIdentity()

        if num_encoder_layers > 0:
            self.encoder = nn.TransformerEncoder(self.encoder_layer(transformer_options), num_encoder_layers)
        else:
            self.encoder = TransformerIdentity()

    @staticmethod
    def encoder_layer(transformer_options: Tuple[int, int, int, float, str]) -> nn.TransformerEncoderLayer:
        """ Encoder layer that will be given to nn.TransformerEncoder.

        This might be overwritten by subclasses.
        """
        return nn.TransformerEncoderLayer(*transformer_options)

    def forward(self, x: Tensor, padding_mask: Tensor, sequence_mask: Tensor) -> Tensor:
        """ Apply time-independent linear layers followed by a transformer encoder.

        This is used during the branches and symmetric attention layers.

        Parameters
        ----------
        x: [T, B, D]
            Input sequence to predict on.
        padding_mask : [B, T]
            Negative mask for transformer input.
        sequence_mask : [T, B, 1]
            Positive mask for zeroing out padded vectors between operations.

        Returns
        -------
        output : [T, B, 1]
            New encoded vectors.
        """
        x = self.embedding(x, sequence_mask)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return x * sequence_mask
