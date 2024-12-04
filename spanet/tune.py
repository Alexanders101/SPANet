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
    import ray
    ray.init()
    from ray import air, tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.hyperopt import HyperOptSearch
    from ray.tune.integration.pytorch_lightning import TuneReportCallback
    from ray.train import RunConfig, ScalingConfig, CheckpointConfig
    from ray.train.lightning import (
            RayDDPStrategy,
            RayFSDPStrategy,
            RayLightningEnvironment,
            RayTrainReportCallback,
            prepare_trainer
    )
    from ray.train.torch import TorchTrainer

except ImportError:
    print("Tuning script requires additional dependencies. Please run: pip install \"ray[tune]\" \"ray[train]\" hyperopt")
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

def get_base_options(base_options_file):
    base_options = Options()
    with open(base_options_file, 'r') as json_file:
        base_options.update_options(json.load(json_file))
    base_options.num_dataloader_workers = 0
    return base_options

def set_spanet_trial(base_options, max_epochs, cpus_per_trial, workers_per_cpu):
    options = base_options
    options.num_dataloader_workers = cpus_per_trial * workers_per_cpu
    num_epochs = max_epochs
    def spanet_trial(config):
        # -------------------------------------------------------------------------------------------------------
        # Create options file and load any optional extra information.
        # -------------------------------------------------------------------------------------------------------
        options.update_options(config)

        # Create base model
        model = JetReconstructionModel(options)

        # Run a simplified trainer for single trial.
        # Typically, we only use 1 gpu per trial so don't need any of the DDP stuff.
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator="auto",
            devices="auto",
            gradient_clip_val=options.gradient_clip if options.gradient_clip > 0 else None,
            enable_progress_bar=False,
            logger=TensorBoardLogger(
                save_dir=os.getcwd(), name="", version="."
            ),
            callbacks=[
                TuneReportCallback(
                    {
                        "loss": "loss/total_loss",
                        "val_avg_accuracy": "validation_average_jet_accuracy"
                    },
                    on="validation_end"
                )
            ],
            strategy=RayDDPStrategy(), 
            plugins=[RayLightningEnvironment()]
        )
        trainer = prepare_trainer(trainer)
        trainer.fit(model)
    return spanet_trial

def tune_spanet(
    base_options_file: str, 
    search_space_file: Optional[str] = None,
    num_trials: int = 10, 
    num_epochs: int = 10, 
    cpus_per_trial: int = 1,
    workers_per_cpu: int = 4, 
    gpus_per_trial: int = 0,
    name: str = "spanet_asha_tune",
    log_dir: str = "spanet_output",
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
        metric_columns=["loss", "val_avg_accuracy", "training_iteration"]
    )
    
    if gpus_per_trial > 0:
        scaling_config = ScalingConfig(
            num_workers=1,
            use_gpu=True,
            resources_per_worker={"CPU": cpus_per_trial, "GPU": gpus_per_trial}
        )
    else:
        scaling_config = ScalingConfig(
            num_workers=1,
            use_gpu=False,
            resources_per_worker={"CPU": cpus_per_trial}
        )

    run_config = air.RunConfig(
        name=name,
        storage_path=log_dir,
        progress_reporter=reporter,
    )
    
    base_options = get_base_options(base_options_file)
    spanet_trial = set_spanet_trial(base_options, num_epochs, cpus_per_trial, workers_per_cpu)

    ray_trainer = TorchTrainer(
        spanet_trial, 
        scaling_config=scaling_config,
        run_config=run_config,
    ) 

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": config},
        tune_config=tune.TuneConfig(
            metric="val_avg_accuracy",
            mode="max",
            scheduler=scheduler,
            search_alg=HyperOptSearch(),
            num_samples=num_trials,
        ),
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
        "-c", "--cpus_per_trial", type=int, default=1,
        help="Number of CPUs available for each parallel trial."
    )

    parser.add_argument(
        "-w", "--workers_per_cpu", type=int, default=4,
        help="Number of dataloader workers per cpu"
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

