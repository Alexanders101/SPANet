# Currently in beta stage, not fully tested.
# Requires
# pip install "ray[tune]==2.5.1" hyperopt

import os
import math

from typing import Optional
from argparse import ArgumentParser
import json

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from spanet import JetReconstructionModel, Options

try:
    from ray import air, tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.hyperopt import HyperOptSearch
    from ray.tune.integration.pytorch_lightning import TuneReportCallback

except ImportError:
    print("Tuning script requires additional dependencies. Please run: pip install \"ray[tune]\" hyperopt")
    exit()


DEFAULT_CONFIG = {
    "hidden_dim": tune.choice([32, 64, 96, 128]),

    "num_encoder_layers": tune.choice([1, 2, 3, 4, 5, 6]),
    "num_branch_embedding_layers": tune.choice([1, 2, 4, 6]),
    "num_branch_encoder_layers": tune.choice([1, 2, 4, 6]),

    "num_regression_layers": tune.choice([1, 2, 4, 6]),
    "num_classification_layers": tune.choice([1, 2, 4, 6]),

    "learning_rate": tune.loguniform(1e-5, 1e-1),
    "focal_gamma": tune.uniform(0.0, 1.0),
    "l2_penalty": tune.loguniform(1e-6, 1e-2)
}

def spanet_trial(config, base_options_file: str, home_dir: str, num_epochs=10, gpus_per_trial: int = 0):
    if not os.path.isabs(base_options_file):
        base_options_file = f"{home_dir}/{base_options_file}"

    # -------------------------------------------------------------------------------------------------------
    # Create options file and load any optional extra information.
    # -------------------------------------------------------------------------------------------------------
    options = Options()
    with open(base_options_file, 'r') as json_file:
        options.update_options(json.load(json_file))

    options.update_options(config)
    options.epochs = num_epochs
    options.num_dataloader_workers = 0

    if not os.path.isabs(options.event_info_file):
        options.event_info_file = f"{home_dir}/{options.event_info_file}"

    if len(options.training_file) > 0 and not os.path.isabs(options.training_file):
        options.training_file = f"{home_dir}/{options.training_file}"

    if len(options.validation_file) > 0 and not os.path.isabs(options.validation_file):
        options.validation_file = f"{home_dir}/{options.validation_file}"

    if len(options.testing_file) > 0 and not os.path.isabs(options.testing_file):
        options.testing_file = f"{home_dir}/{options.testing_file}"

    # Create base model
    model = JetReconstructionModel(options)

    # Run a simplified trainer for single trial.
    # Typically, we only use 1 gpu per trial so don't need any of the DDP stuff.
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu" if gpus_per_trial > 0 else None,
        devices=gpus_per_trial if gpus_per_trial > 0 else None,
        gradient_clip_val=options.gradient_clip if options.gradient_clip > 0 else None,
        enable_progress_bar=False,
        logger=TensorBoardLogger(
            save_dir=os.getcwd(), name="", version="."
        ),
        callbacks=[
            TuneReportCallback(
                {
                    "loss": "loss/total_loss",
                    "mean_accuracy": "validation_accuracy"
                },
                on="validation_end"
            )
        ])
    trainer.fit(model)


def tune_spanet(
    base_options_file: str, 
    search_space_file: Optional[str] = None,
    num_trials: int = 10, 
    num_epochs: int = 10, 
    gpus_per_trial: int = 0,
    name: str = "spanet_asha_tune",
    log_dir: str = "spanet_output"
):
    # Load the search space. 
    # This seems to be the best way to load arbitrary tune search spaces.
    # Not great due to the dynamic eval but ray doesnt have a config spec for search spaces.
    config = DEFAULT_CONFIG
    if search_space_file is not None:
        with open(search_space_file, 'r') as file:
            search_space = json.load(file)
        
        config = {
            key: eval(value) if isinstance(value, str) and ("tune." in value) else value
            for key, value in search_space.items()
        }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=num_epochs //  4,
        reduction_factor=2
    )

    reporter = CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=["loss", "mean_accuracy", "training_iteration"]
    )

    train_fn_with_parameters = tune.with_parameters(
        spanet_trial,
        base_options_file=base_options_file,
        home_dir=os.getcwd(),
        num_epochs=num_epochs,
        gpus_per_trial=gpus_per_trial
    )

    resources_per_trial = {"cpu": 1, "gpu": gpus_per_trial}

    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources=resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            search_alg=HyperOptSearch(),
            num_samples=num_trials,
        ),
        run_config=air.RunConfig(
            name=name,
            storage_path=log_dir,
            progress_reporter=reporter,
        ),
        param_space=config,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        "base_options_file", type=str,
        help="Base options file to load and adjust with tuning parameters."
    )

    parser.add_argument(
        "-s", "--search_space_file", type=str, default=None,
        help="JSON file with tune search space definitions to override default."
    )

    parser.add_argument(
        "-g", "--gpus_per_trial", type=int, default=0,
        help="Number of GPUs available for each parallel trial."
    )

    parser.add_argument(
        "-e", "--num_epochs", type=int, default=128,
        help="Number of training epochs per trial"
    )

    parser.add_argument(
        "-t", "--num_trials", type=int, default=10,
        help="Number of trials to run."
    )

    parser.add_argument(
        "-l", "--log_dir", type=str, default="spanet_output",
        help="Output directory for all trials.")

    parser.add_argument(
        "-n", "--name", type=str, default="spanet_asha_tune",
        help="The sub-directory to create for this run.")

    tune_spanet(**parser.parse_args().__dict__)

