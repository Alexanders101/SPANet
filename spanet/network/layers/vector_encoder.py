from spanet.options import Options
from spanet.network.layers.stacked_encoder import StackedEncoder


class JetEncoder(StackedEncoder):
    def __init__(self, options: Options):
        super(JetEncoder, self).__init__(options, 0, options.num_encoder_layers)
