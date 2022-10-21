from torch import Tensor, nn, jit

from spanet.options import Options
from spanet.network.layers.linear_block.gru_block import GRUGate, GRUBlock
from spanet.network.layers.transformer.transformer_base import TransformerBase


class GTrXL(nn.Module):
    def __init__(self, options, hidden_dim: int, num_heads: int, dropout: float):
        super(GTrXL, self).__init__()

        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention_gate = GRUGate(hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.feed_forward = GRUBlock(options, hidden_dim, hidden_dim, skip_connection=True)

    def forward(self, x: Tensor, padding_mask: Tensor, sequence_mask: Tensor) -> Tensor:
        output = self.attention_norm(x)
        output, _ = self.attention(
            output, output, output,
            key_padding_mask=padding_mask,
            need_weights=False
        )

        output = self.attention_gate(output, x)

        return self.feed_forward(output, sequence_mask)


class GatedTransformer(TransformerBase):
    def __init__(self, options: Options, num_layers: int):
        super(GatedTransformer, self).__init__(options, num_layers)

        self.layers = nn.ModuleList([
            GTrXL(options, self.hidden_dim, self.num_heads, self.dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor, padding_mask: Tensor, sequence_mask: Tensor) -> Tensor:
        output = x

        for layer in self.layers:
            output = layer(output, padding_mask, sequence_mask)

        return output
