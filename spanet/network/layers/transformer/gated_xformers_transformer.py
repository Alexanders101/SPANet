from typing import Optional
from torch import Tensor, nn

from spanet.options import Options
from spanet.network.layers.transformer.transformer_base import TransformerBase
from spanet.network.layers.linear_block.masking import create_masking

XFORMERS_LOADED = False
try:
    import xformers
    from xformers.components import attention, feedforward, Activation, PreNorm, MultiHeadDispatch, NormalizationType
    from xformers.triton import FusedDropoutBias, FusedLayerNorm

    XFORMERS_LOADED = True
except ImportError:
    XFORMERS_LOADED = False


class FusedMLP(nn.Sequential):

    def __init__(
            self,
            dim_model: int,
            dim_mlp: int,
            dropout: float,
            activation: Activation,
            bias: bool = True,
    ):
        super(FusedMLP, self).__init__(
            nn.Linear(in_features=dim_model, out_features=dim_mlp, bias=False),
            FusedDropoutBias(
                p=dropout,
                bias_shape=dim_mlp if bias else None,
                activation=activation,
            ),
            nn.Linear(in_features=dim_mlp, out_features=dim_model, bias=False),
            FusedDropoutBias(
                p=dropout,
                bias_shape=dim_model if bias else None,
                activation=None,
            )
        )


class WrappedAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout):
        super(WrappedAttention, self).__init__(embed_dim, num_heads, dropout)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        return super(WrappedAttention, self).forward(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )[0]


class XFormersGTrXL(nn.Module):
    __constants__ = ["hidden_dim", "mlp_dim"]

    def __init__(self, options, hidden_dim: int, num_heads: int, dropout: float):
        super(XFormersGTrXL, self).__init__()

        self.hidden_dim = hidden_dim
        self.mlp_dim = int(round(options.transformer_dim_scale * hidden_dim))

        self.attention_norm = FusedLayerNorm(hidden_dim)
        self.attention = WrappedAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.feedforward = nn.Sequential(
            FusedLayerNorm(
                hidden_dim
            ),
            FusedMLP(
                dim_model=self.hidden_dim,
                dim_mlp=self.mlp_dim,
                dropout=dropout,
                activation=Activation.GeLU,
            )
        )

        self.attention_gate = nn.GRUCell(hidden_dim, hidden_dim)
        self.feedforward_gate = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x: Tensor, padding_mask: Tensor) -> Tensor:
        shape = x.shape

        outputs = self.attention(self.attention_norm(x), key_padding_mask=padding_mask)

        outputs = outputs.view(-1, self.hidden_dim)
        outputs = self.attention_gate(outputs, x.view(-1, self.hidden_dim))

        return self.feedforward_gate(self.feedforward(outputs), outputs).view(shape)


class XFormersGatedTransformer(TransformerBase):
    def __init__(self, options: Options, num_layers: int):
        super(XFormersGatedTransformer, self).__init__(options, num_layers)

        self.masking = create_masking(options.masking)

        self.layers = nn.ModuleList([
            XFormersGTrXL(options, self.hidden_dim, self.num_heads, self.dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor, padding_mask: Tensor, sequence_mask: Tensor) -> Tensor:
        output = x

        for layer in self.layers:
            output = layer(output, padding_mask)

        return self.masking(output, sequence_mask)
