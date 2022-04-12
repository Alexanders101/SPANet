from torch import Tensor, nn

from spanet.options import Options


class TransformerBase(nn.Module):
    __constants__ = ["num_layers", "hidden_dim", "num_heads", "dim_feedforward", "dropout", "transformer_activation"]

    def __init__(self, options: Options, num_layers: int):
        super(TransformerBase, self).__init__()

        self.num_layers = num_layers

        self.dropout = options.dropout
        self.hidden_dim = options.hidden_dim
        self.num_heads = options.num_attention_heads
        self.transformer_activation = options.transformer_activation
        self.dim_feedforward = int(round(options.transformer_dim_scale * options.hidden_dim))

    def forward(self, x: Tensor, padding_mask: Tensor, sequence_mask: Tensor) -> Tensor:
        return x
