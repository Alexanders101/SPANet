from argparse import ArgumentParser
from typing import Optional
from os import getcwd, makedirs, environ
import shutil
import json

import torch
import pytorch_lightning as pl
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress.rich_progress import _RICH_AVAILABLE
from pytorch_lightning.loggers.wandb import _WANDB_AVAILABLE, WandbLogger

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary,
    DeviceStatsMonitor,
    ModelSummary,
    TQDMProgressBar
)

from spanet import JetReconstructionModel, Options


def main(
        event_file: str,
        training_file: str,
        validation_file: str,
        options_file: Optional[str],
        checkpoint: Optional[str],
        state_dict: Optional[str],
        freeze_state_dict: bool,

        log_dir: str,
        name: str,

        torch_script: bool,
        fp16: bool,
        verbose: bool,
        full_events: bool,

        profile: bool,
        gpus: Optional[int],
        epochs: Optional[int],
        time_limit: Optional[str],
        batch_size: Optional[int],
        limit_dataset: Optional[float],
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
    model = JetReconstructionModel(options, torch_script)

    if state_dict is not None:
        if master:
            print(f"Loading state dict from: {state_dict}")

        state_dict = torch.load(state_dict, map_location="cpu")["state_dict"]
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if master:
            print(f"Missing Keys: {missing_keys}")
            print(f"Unexpected Keys: {unexpected_keys}")

        if freeze_state_dict:
            for pname, parameter in model.named_parameters():
                if pname in state_dict:
                    parameter.requires_grad_(False)

    # Construct the logger for this training run. Logs will be saved in {logdir}/{name}/version_i
    log_dir = getcwd() if log_dir is None else log_dir
    logger = (
        WandbLogger(name=name, save_dir=log_dir)
        if _WANDB_AVAILABLE else
        TensorBoardLogger(save_dir=log_dir, name=name)
    )

    # Create the checkpoint for this training run. We will save the best validation networks based on 'accuracy'
    callbacks = [
        ModelCheckpoint(
            verbose=options.verbose_output,
            monitor='validation_accuracy',
            save_top_k=3,
            mode='max',
            save_last=True
        ),
        LearningRateMonitor(),
        DeviceStatsMonitor(),
        RichProgressBar() if _RICH_AVAILABLE else TQDMProgressBar(),
        RichModelSummary(max_depth=1) if _RICH_AVAILABLE else ModelSummary(max_depth=1)
    ]

    epochs = options.epochs
    profiler = None
    if profile:
        epochs = 1
        profiler = PyTorchProfiler(emit_nvtx=True)

    # Create the final pytorch-lightning manager
    trainer = pl.Trainer(
        accelerator="gpu" if options.num_gpu > 0 else "auto",
        devices=options.num_gpu if options.num_gpu > 0 else "auto",
        strategy="ddp" if options.num_gpu > 1 else "auto",
        precision="16-mixed" if fp16 else "32-true",

        gradient_clip_val=options.gradient_clip if options.gradient_clip > 0 else None,
        max_epochs=epochs,
        max_time=time_limit,

        logger=logger,
        profiler=profiler,
        callbacks=callbacks
    )

    # Save the current hyperparameters to a json file in the checkpoint directory
    if master:
        print(f"Training Version {trainer.logger.version}")
        makedirs(trainer.logger.log_dir, exist_ok=True)

        with open(f"{trainer.logger.log_dir}/options.json", 'w') as json_file:
            json.dump(options.__dict__, json_file, indent=4)

        shutil.copy2(options.event_info_file, f"{trainer.logger.log_dir}/event.yaml")

    trainer.fit(model, ckpt_path=checkpoint)
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
                        help="Optional checkpoint to load the training state from. "
                             "Fully restores model weights and optimizer state.")

    parser.add_argument("-sf", "--state_dict", type=str, default=None,
                        help="Load from checkpoint but only the model weights. "
                             "Can be partial as the weights don't have to match one-to-one.")

    parser.add_argument("-fsf", "--freeze_state_dict", action='store_true',
                        help="Freeze any weights that were loaded from the state dict. "
                             "Used for finetuning new layers.")

    parser.add_argument("-l", "--log_dir", type=str, default=None,
                        help="Output directory for the checkpoints and tensorboard logs. Default to current directory.")

    parser.add_argument("-n", "--name", type=str, default="spanet_output",
                        help="The sub-directory to create for this run and an identifier for WANDB.")

    parser.add_argument("-e", "--epochs", type=int, default=None,
                        help="Override number of epochs to train for")
    
    parser.add_argument("-t", "--time_limit", type=str, default=None,
                        help="Time limit for training, in the format DD:HH:MM:SS.")

    parser.add_argument("-g", "--gpus", type=int, default=None,
                        help="Override GPU count in hyperparameters.")
    
    parser.add_argument("-b", "--batch_size", type=int, default=None,
                        help="Override batch size in hyperparameters.")

    parser.add_argument("-f", "--full_events", action='store_true',
                        help="Limit training to only full events.")

    parser.add_argument("-p", "--limit_dataset", type=float, default=None,
                        help="Limit dataset to only the first L percent of the data (0 - 100).")

    parser.add_argument("-fp16", "--fp16", action="store_true",
                        help="Use Torch AMP for training.")

    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Output additional information to console and log.")

    parser.add_argument("-r", "--random_seed", type=int, default=0,
                        help="Set random seed for cross-validation.")

    parser.add_argument("-ts", "--torch_script", action='store_true',
                        help="Compile the neural network using torchscript.")

    parser.add_argument("--profile", action='store_true',
                        help="Profile network for a single training epoch.")

    main(**parser.parse_args().__dict__)