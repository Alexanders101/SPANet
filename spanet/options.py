import json
from argparse import Namespace


class Options(Namespace):
    def __init__(self, event_info_file: str = "", training_file: str = "", validation_file: str = "", testing_file: str = ""):
        super(Options, self).__init__()

        # =========================================================================================
        # Network Architecture
        # =========================================================================================

        # Dimensions used internally by all hidden layers / transformers.
        self.hidden_dim: int = 128

        # DEPRECATED
        # Internal dimensions used during transformer and some linear layers.
        self.transformer_dim: int = 128

        # Internal dimensions used during transformer and some linear layers.
        # Scalar variant, muiltiply the input dim by this amount.
        self.transformer_dim_scale: float = 2.0

        # Hidden dimensionality of the first embedding layer.
        self.initial_embedding_dim: int = 16

        self.position_embedding_dim: int = 32

        # Maximum Number of double-sized embedding layers to add between the features and the encoder.
        # The size of the embedding dimension will be capped at the hidden_dim,
        # So setting this option to a very large integer will just keep embedding up to the hidden_dim.
        self.num_embedding_layers: int = 10

        # Number of encoder layers for the central shared transformer.
        self.num_encoder_layers: int = 4

        # Number of feed forward layers to add to branch heads.
        # Set to 0 to disable branch embedding layers.
        self.num_branch_embedding_layers: int = 4

        # Number of encoder layers for each of the quark branch transformers.
        # Set to 0 to disable branch encoder layers.
        self.num_branch_encoder_layers: int = 4

        # Number of extra linear layers before the attention layer when using split attention.
        # Only used if split_symmetric_attention is True
        # Set to 0 to disable jet embedding layers.
        self.num_jet_embedding_layers: int = 0

        # Number of extra transformer layers before the attention layer when using split attention.
        # Only used if split_symmetric_attention is True
        # Set to 0 to disable jet encoder layers.
        self.num_jet_encoder_layers: int = 0

        # Number of hidden layers to use for the particle classification head.
        self.num_detector_layers: int = 1

        # Number of hidden layers to use for the particle classification head.
        self.num_regression_layers: int = 1

        # Number of hidden layers to use for the particle classification head.
        self.num_classification_layers: int = 1

        # Whether or not to use a split approximate tensor attention layer.
        self.split_symmetric_attention: bool = True

        # Number of heads for multi-head attention, used in all transformer layers.
        self.num_attention_heads: int = 4

        # Activation function for all transformer layers, 'relu' or 'gelu'.
        self.transformer_activation: str = 'gelu'

        # Whether or not to add skip connections to internal linear layers.
        # All layers support skip connections, this can turn them off.
        self.skip_connections: bool = True

        # Whether or not to add skip connections to the initial set of embedding layers.
        self.initial_embedding_skip_connections: bool = True

        # Structure for linear layers in the network
        #
        # Options are:
        # -------------------------------------------------
        # Basic
        # Resnet
        # Gated
        # GRU
        # -------------------------------------------------
        self.linear_block_type: str = "GRU"

        # Structure for transformer layer
        #
        # Options are:
        # -------------------------------------------------
        # Standard
        # NormFirst
        # Gated
        # -------------------------------------------------
        self.transformer_type: str = "Gated"

        # Non-linearity to use inside of the linear blocks.
        #
        # Options are:
        # -------------------------------------------------
        # None
        # ReLU
        # PReLU
        # ELU
        # GELU
        # -------------------------------------------------
        self.linear_activation: str = "GELU"

        # Whether or not to apply a normalization layer during linear / embedding layers.
        #
        # Options are:
        # -------------------------------------------------
        # None
        # BatchNorm
        # LayerNorm
        # MaskedBatchNorm
        # -------------------------------------------------
        self.normalization: str = "LayerNorm"

        # What type of masking to use throughout the linear layers.
        #
        # Options are:
        # -------------------------------------------------
        # None
        # Multiplicative
        # Filling
        # -------------------------------------------------
        self.masking: str = "Filling"

        # DEPRECATED
        # Whether or not to use PreLU activation on linear / embedding layers,
        # Otherwise a regular relu will be used.
        self.linear_prelu_activation: bool = True

        # =========================================================================================
        # Dataset Options
        # =========================================================================================

        # Location of event ini file and the jet hdf5 files.
        # This is set by the constructor and should not be set manually.
        self.event_info_file: str = event_info_file
        self.training_file: str = training_file
        self.validation_file: str = validation_file
        self.testing_file: str = testing_file

        # Whether or not to compute training data statistics to normalize features.
        self.normalize_features: bool = True

        # Limit the dataset to this exact number of jets. Set to 0 to disable.
        self.limit_to_num_jets: int = 0

        # Whether or not to add weight to classes based on their training data prevalence.
        self.balance_particles: bool = False

        # Whether or not to add a weight to the jet multiplicity to not forget about large events.
        self.balance_jets: bool = False

        # Whether or not to add a weight to classification heads based on target presence.
        self.balance_classifications: bool = False

        # Whether or not to train on partial events in the dataset.
        self.partial_events: bool = False

        # Limit the dataset to the first x% of the data.
        self.dataset_limit: float = 1.0

        # Set a non-zero value here to deterministically shuffle the training dataset when selecting the subset.
        self.dataset_randomization: int = 0

        # Percent of data to use for training vs. validation.
        self.train_validation_split: float = 0.95

        # Training batch size.
        self.batch_size: int = 4096

        # Number of processes to spawn for data collection.
        self.num_dataloader_workers: int = 4

        # =========================================================================================
        # Training Options
        # =========================================================================================

        # Whether or not to mask vectors not in the events during operation.
        # Should most-definitely be True, but this is here for testing.
        self.mask_sequence_vectors: bool = True

        # Whether we should combine the two possible targets: swapped and not-swapped.
        # If None, then we will only use the proper target ordering.
        #
        # Options are:
        # -------------------------------------------------
        # None
        # min
        # softmin
        # mean
        # -------------------------------------------------
        self.combine_pair_loss: str = 'min'

        # The optimizer to use for trianing the network.
        # This must be a valid class in torch.optim or nvidia apex with 'apex' prefix.
        self.optimizer: str = "AdamW"

        # Optimizer learning rate.
        self.learning_rate: float = 0.001

        # Gamma exponent for focal loss. Setting it to 0.0 will disable focal loss and use regular cross-entropy.
        self.focal_gamma: float = 0.0

        # Combinatorial offset for the masked softmax discrepancy
        self.combinatorial_scale: float = 0.0

        # Number of epochs to ramp up the learning rate up to the given value. Can be fractional.
        self.learning_rate_warmup_epochs: float = 1.0

        # Number of times to cycles the learning rate through cosine annealing with hard resets.
        # Set to 0 to disable cosine annealing and just use a decaying learning rate.
        self.learning_rate_cycles: int = 0

        # Scalar term for the primary jet assignment loss.
        self.assignment_loss_scale: float = 1.0

        # Scalar term for the direct classification loss of particles.
        self.detection_loss_scale: float = 0.0

        # Scalar term for the symmetric KL-divergence loss between distributions.
        self.kl_loss_scale: float = 0.0

        # Scalar term for regression L2 loss term
        self.regression_loss_scale: float = 0.0

        # Scalar term for classification Cross Entropy loss term
        self.classification_loss_scale: float = 0.0

        # Automatically balance loss terms using Jacobians.
        self.balance_losses: bool = True

        # Optimizer l2 penalty based on weight values.
        self.l2_penalty: float = 0.0

        # Clip the L2 norm of the gradient. Set to 0.0 to disable.
        self.gradient_clip: float = 0.0

        # Dropout added to all layers.
        self.dropout: float = 0.0

        # Number of epochs to train for.
        self.epochs: int = 100
        
        # Total number of GPUs to use.
        self.num_gpu: int = 1

        # =========================================================================================
        # Miscellaneous Options
        # =========================================================================================

        # Whether or not to print additional information during training and log extra metrics.
        self.verbose_output: bool = True

        # Misc parameters used by sherpa to delegate GPUs and output directories.
        # These should not be set manually.
        self.usable_gpus: str = ''

        self.trial_time: str = ''

        self.trial_output_dir: str = './test_output'

    def display(self):
        try:
            from rich import get_console
            from rich.table import Table

            default_options = self.__class__().__dict__
            console = get_console()

            table = Table(title="Configuration", header_style="bold magenta")
            table.add_column("Parameter", justify="left")
            table.add_column("Value", justify="left")

            for key, value in sorted(self.__dict__.items()):
                table.add_row(key, str(value), style="red" if value != default_options[key] else None)

            console.print(table)

        except ImportError:
            print("=" * 70)
            print("Options")
            print("-" * 70)
            for key, val in sorted(self.__dict__.items()):
                print(f"{key:32}: {val}")
            print("=" * 70)

    def update_options(self, new_options, update_datasets: bool = True):
        integer_options = {key for key, val in self.__dict__.items() if isinstance(val, int)}
        boolean_options = {key for key, val in self.__dict__.items() if isinstance(val, bool)}
        for key, value in new_options.items():
            if not update_datasets and key in {"event_info_file", "training_file", "validation_file", "testing_file"}:
                continue

            if key in integer_options:
                setattr(self, key, int(value))
            elif key in boolean_options:
                setattr(self, key, bool(value))
            else:
                setattr(self, key, value)

    def update(self, filepath: str):
        with open(filepath, 'r') as json_file:
            self.update_options(json.load(json_file))

    @classmethod
    def load(cls, filepath: str):
        options = cls()
        with open(filepath, 'r') as json_file:
            options.update_options(json.load(json_file))
        return options

    def save(self, filepath: str):
        with open(filepath, 'w') as json_file:
            json.dump(self.__dict__, json_file, indent=4, sort_keys=True)
