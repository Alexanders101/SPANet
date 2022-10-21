from torch import Tensor, nn

from spanet.options import Options
from spanet.network.layers.linear_block.masking import create_masking
from spanet.network.layers.transformer.transformer_base import TransformerBase


class NormFirstTransformer(TransformerBase):
    def __init__(self, options: Options, num_layers: int):
        super(NormFirstTransformer, self).__init__(options, num_layers)

        self.masking = create_masking(options.masking)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.hidden_dim,
                self.num_heads,
                self.dim_feedforward,
                self.dropout,
                self.transformer_activation,
                norm_first=True
            ),
            num_layers
        )

    def forward(self, x: Tensor, padding_mask: Tensor, sequence_mask: Tensor) -> Tensor:
        output = self.transformer(x, src_key_padding_mask=padding_mask)
        return self.masking(output, sequence_mask)
