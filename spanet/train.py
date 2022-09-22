from argparse import ArgumentParser
from typing import Optional
from os import getcwd, makedirs, environ
import json

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary

from spanet import JetReconstructionModel, Options


def main(event_file: str,
         training_file: str,
         validation_file: str,
         options_file: Optional[str],
         checkpoint: Optional[str],

         log_dir: str,
         name: str,

         fp16: bool,
         graph: bool,
         verbose: bool,
         full_events: bool,

         gpus: Optional[int],
         epochs: Optional[int],
         batch_size: Optional[int],
         limit_dataset: Optional[int],
         random_seed: int,
         ):

    # Whether or not this script version is the master run or a worker
    master = True
    if "NODE_RANK" in environ:
        master = False

    # -------------------------------------------------------------------------------------------------------
    # Create options file and load any optional extra information.
    # -------------------------------------------------------------------------------------------------------
    options = Options(event_file, training_file, validation_file)

    if options_file is not None:
        with open(options_file, 'r') as json_file:
            options.update_options(json.load(json_file))

    # -------------------------------------------------------------------------------------------------------
    # Command line overrides for common option values.
    # -------------------------------------------------------------------------------------------------------
    options.verbose_output = verbose
    if master and verbose:
        print(f"Verbose output activated.")

    if full_events:
        if master:
            print(f"Overriding: Only using full events")
        options.partial_events = False
        options.balance_particles = False

    if gpus is not None:
        if master:
            print(f"Overriding GPU count: {gpus}")
        options.num_gpu = gpus

    if batch_size is not None:
        if master:
            print(f"Overriding Batch Size: {batch_size}")
        options.batch_size = batch_size

    if limit_dataset is not None:
        if master:
            print(f"Overriding Dataset Limit: {limit_dataset}%")
        options.dataset_limit = limit_dataset / 100

    if epochs is not None:
        if master:
            print(f"Overriding Number of Epochs: {epochs}")
        options.epochs = epochs

    if random_seed > 0:
        options.dataset_randomization = random_seed

    # -------------------------------------------------------------------------------------------------------
    # Print the full hyperparameter list
    # -------------------------------------------------------------------------------------------------------
    if master:
        options.display()

    # -------------------------------------------------------------------------------------------------------
    # Begin the training loop
    # -------------------------------------------------------------------------------------------------------

    # Create the initial model on the CPU
    model = JetReconstructionModel(options)

    # If we are using more than one gpu, then switch to DDP training
    strategy = 'dp' if options.num_gpu > 1 else None
    # strategy = 'ddp' if options.num_gpu > 1 else None

    # Construct the logger for this training run. Logs will be saved in {logdir}/{name}/version_i
    log_dir = getcwd() if log_dir is None else log_dir
    logger = TensorBoardLogger(save_dir=log_dir, name=name, log_graph=graph)

    # Create the checkpoint for this training run. We will save the best validation networks based on 'accuracy'
    checkpoint_callback = ModelCheckpoint(verbose=options.verbose_output,
                                          monitor='validation_accuracy',
                                          save_top_k=1,
                                          mode='max',
                                          save_last=True)

    learning_rate_callback = LearningRateMonitor()

    model_summary = ModelSummary(max_depth=10 if options.verbose_output else 1)

    # Create the final pytorch-lightning manager
    trainer = pl.Trainer(logger=logger,
                         max_epochs=options.epochs,
                         callbacks=[checkpoint_callback, learning_rate_callback, model_summary],
                         resume_from_checkpoint=checkpoint,
                         strategy=strategy,
                         accelerator='gpu' if options.num_gpu > 0 else 'cpu',
                         devices=options.num_gpu if options.num_gpu > 0 else None,
                         track_grad_norm=2 if options.verbose_output else -1,
                         gradient_clip_val=options.gradient_clip,
                         precision=16 if fp16 else 32)

    # Save the current hyperparameters to a json file in the checkpoint directory
    if master:
        print(f"Training Version {trainer.logger.version}")
        makedirs(trainer.logger.log_dir, exist_ok=True)
        with open(trainer.logger.log_dir + "/options.json", 'w') as json_file:
            json.dump(options.__dict__, json_file, indent=4)

    trainer.fit(model)
    # -------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-ef", "--event_file", type=str, default="",
                        help="Input file containing event symmetry information.")

    parser.add_argument("-tf", "--training_file", type=str, default="",
                        help="Input file containing training data.")

    parser.add_argument("-vf", "--validation_file", type=str, default="",
                        help="Input file containing Validation data. If not provided, will use training data split.")

    parser.add_argument("-of", "--options_file", type=str, default=None,
                        help="JSON file with option overloads.")

    parser.add_argument("-cf", "--checkpoint", type=str, default=None,
                        help="Optional checkpoint to load from")

    parser.add_argument("-l", "--log_dir", type=str, default=None,
                        help="Output directory for the checkpoints and tensorboard logs. Default to current directory.")

    parser.add_argument("-n", "--name", type=str, default="spanet_output",
                        help="The sub-directory to create for this run.")

    parser.add_argument("-fp16", action="store_true",
                        help="Use AMP for training.")

    parser.add_argument("-g", "--graph", action="store_true",
                        help="Log the computation graph.")

    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Output additional information to console and log.")

    parser.add_argument("-b", "--batch_size", type=int, default=None,
                        help="Override batch size in hyperparameters.")

    parser.add_argument("-f", "--full_events", action='store_true',
                        help="Limit training to only full events.")

    parser.add_argument("-p", "--limit_dataset", type=int, default=None,
                        help="Limit dataset to only the first L percent of the data (0 - 100).")

    parser.add_argument("-r", "--random_seed", type=int, default=0,
                        help="Set random seed for cross-validation.")

    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs to train for")

    parser.add_argument("--gpus", type=int, default=None,
                        help="Override GPU count in hyperparameters.")

    main(**parser.parse_args().__dict__)
