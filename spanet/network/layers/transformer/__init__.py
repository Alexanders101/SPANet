from torch import jit

from spanet.options import Options

from spanet.network.layers.transformer.transformer_base import TransformerBase
from spanet.network.layers.transformer.gated_transformer import GatedTransformer
from spanet.network.layers.transformer.standard_transformer import StandardTransformer
from spanet.network.layers.transformer.norm_first_transformer import NormFirstTransformer


def create_transformer(
        options: Options,
        num_layers: int,
):
    transformer_type = options.transformer_type
    transformer_type = transformer_type.lower().replace("_", "").replace(" ", "")

    if num_layers <= 0:
        return jit.script(TransformerBase(options, num_layers))

    if transformer_type == "standard":
        return jit.script(StandardTransformer(options, num_layers))
    elif transformer_type == 'normfirst':
        return jit.script(NormFirstTransformer(options, num_layers))
    elif transformer_type == 'gated':
        return jit.script(GatedTransformer(options, num_layers))
    elif transformer_type == 'gtrxl':
        return jit.script(GatedTransformer(options, num_layers))
    else:
        return jit.script(TransformerBase(options, num_layers))
