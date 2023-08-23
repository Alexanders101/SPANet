import pytorch_lightning as pl
import numpy as np
import torch
from torch import nn

# noinspection PyProtectedMember
from torch.utils.data import DataLoader

from spanet.options import Options
from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset
from spanet.network.learning_rate_schedules import get_linear_schedule_with_warmup
from spanet.network.learning_rate_schedules import get_cosine_with_hard_restarts_schedule_with_warmup


class JetReconstructionBase(pl.LightningModule):
    def __init__(self, options: Options):
        super(JetReconstructionBase, self).__init__()

        self.save_hyperparameters(options)
        self.options = options

        self.training_dataset, self.validation_dataset, self.testing_dataset = self.create_datasets()

        # Compute class weights for particles from the training dataset target distribution
        self.balance_particles = False
        if options.balance_particles and options.partial_events:
            index_tensor, weights_tensor = self.training_dataset.compute_particle_balance()
            self.particle_index_tensor = torch.nn.Parameter(index_tensor, requires_grad=False)
            self.particle_weights_tensor = torch.nn.Parameter(weights_tensor, requires_grad=False)
            self.balance_particles = True

        # Compute class weights for jets from the training dataset target distribution
        self.balance_jets = False
        if options.balance_jets:
            jet_weights_tensor = self.training_dataset.compute_vector_balance()
            self.jet_weights_tensor = torch.nn.Parameter(jet_weights_tensor, requires_grad=False)
            self.balance_jets = True

        self.balance_classifications = options.balance_classifications
        if self.balance_classifications:
            classification_weights = {
                key: torch.nn.Parameter(value, requires_grad=False)
                for key, value in self.training_dataset.compute_classification_balance().items()
            }

            self.classification_weights = torch.nn.ParameterDict(classification_weights)

        # Helper arrays for permutation groups. Used for the partial-event loss functions.
        event_permutation_group = np.array(self.event_info.event_permutation_group)
        self.event_permutation_tensor = torch.nn.Parameter(torch.from_numpy(event_permutation_group), False)

        # Helper variables for keeping track of the number of batches in each epoch.
        # Used for learning rate scheduling and other things.
        self.steps_per_epoch = len(self.training_dataset) // (self.options.batch_size * max(1, self.options.num_gpu))
        # self.steps_per_epoch = len(self.training_dataset) // self.options.batch_size
        self.total_steps = self.steps_per_epoch * self.options.epochs
        self.warmup_steps = int(round(self.steps_per_epoch * self.options.learning_rate_warmup_epochs))

    @property
    def dataset(self):
        return JetReconstructionDataset

    @property
    def dataloader(self):
        return DataLoader

    @property
    def dataloader_options(self):
        return {
            "batch_size": self.options.batch_size,
            "pin_memory": self.options.num_gpu > 0,
            "num_workers": self.options.num_dataloader_workers,
        }

    @property
    def event_info(self):
        return self.training_dataset.event_info

    def create_datasets(self):
        event_info_file = self.options.event_info_file
        training_file = self.options.training_file
        validation_file = self.options.validation_file

        training_range = self.options.dataset_limit
        validation_range = 1.0

        # If we dont have a validation file provided, create one from the training file.
        if len(validation_file) == 0:
            validation_file = training_file

            # Compute the training / validation ranges based on the data-split and the limiting percentage.
            train_validation_split = self.options.dataset_limit * self.options.train_validation_split
            training_range = (0.0, train_validation_split)
            validation_range = (train_validation_split, self.options.dataset_limit)

        # Construct primary training datasets
        # Note that only the training dataset should be limited to full events or partial events.
        training_dataset = self.dataset(
            data_file=training_file,
            event_info=event_info_file,
            limit_index=training_range,
            vector_limit=self.options.limit_to_num_jets,
            partial_events=self.options.partial_events,
            randomization_seed=self.options.dataset_randomization
        )

        validation_dataset = self.dataset(
            data_file=validation_file,
            event_info=event_info_file,
            limit_index=validation_range,
            vector_limit=self.options.limit_to_num_jets,
            randomization_seed=self.options.dataset_randomization
        )

        # Optionally construct the testing dataset.
        # This is not used in the main training script but is still useful for testing later.
        testing_dataset = None
        if len(self.options.testing_file) > 0:
            testing_dataset = self.dataset(
                data_file=self.options.testing_file,
                event_info=self.options.event_info_file,
                limit_index=1.0,
                vector_limit=self.options.limit_to_num_jets
            )

        return training_dataset, validation_dataset, testing_dataset

    def configure_optimizers(self):
        optimizer = None

        if 'apex' in self.options.optimizer:
            try:
                # noinspection PyUnresolvedReferences
                import apex.optimizers

                if self.options.optimizer == 'apex_adam':
                    optimizer = apex.optimizers.FusedAdam

                elif self.options.optimizer == 'apex_lamb':
                    optimizer = apex.optimizers.FusedLAMB

                else:
                    optimizer = apex.optimizers.FusedSGD

            except ImportError:
                pass

        else:
            optimizer = getattr(torch.optim, self.options.optimizer)

        if optimizer is None:
            print(f"Unable to load desired optimizer: {self.options.optimizer}.")
            print(f"Using pytorch AdamW as a default.")
            optimizer = torch.optim.AdamW

        decay_mask = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [param for name, param in self.named_parameters()
                           if not any(no_decay in name for no_decay in decay_mask)],
                "weight_decay": self.options.l2_penalty,
            },
            {
                "params": [param for name, param in self.named_parameters()
                           if any(no_decay in name for no_decay in decay_mask)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = optimizer(optimizer_grouped_parameters, lr=self.options.learning_rate)

        if self.options.learning_rate_cycles < 1:
            scheduler = get_linear_schedule_with_warmup(
                 optimizer,
                 num_warmup_steps=self.warmup_steps,
                 num_training_steps=self.total_steps
             )
        else:
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps,
                num_cycles=self.options.learning_rate_cycles
            )

        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler]

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(self.training_dataset, shuffle=True, drop_last=True, **self.dataloader_options)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(self.validation_dataset, drop_last=True, **self.dataloader_options)

    def test_dataloader(self) -> DataLoader:
        if self.testing_dataset is None:
            raise ValueError("Testing dataset not provided.")

        return self.dataloader(self.testing_dataset, **self.dataloader_options)
