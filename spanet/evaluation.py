from glob import glob
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from spanet import JetReconstructionModel, Options


def load_model(log_directory: str,
               testing_file: Optional[str] = None,
               event_info_file: Optional[str] = None,
               batch_size: Optional[int] = None,
               cuda: bool = False) -> JetReconstructionModel:
    # Load the best-performing checkpoint on validation data
    checkpoint = sorted(glob(f"{log_directory}/checkpoints/epoch*"))[-1]
    checkpoint = torch.load(checkpoint, map_location='cpu')
    checkpoint = checkpoint["state_dict"]

    # Load the options that were used for this run and set the testing-dataset value
    options = Options.load(f"{log_directory}/options.json")

    # Override options from command line arguments
    if testing_file is not None:
        options.testing_file = testing_file

    if event_info_file is not None:
        options.event_info_file = event_info_file

    if batch_size is not None:
        options.batch_size = batch_size

    # We need a testing file defined somewhere to continue
    if options.testing_file is None or options.testing_file == "":
        raise ValueError("No testing file found in model options or provided to test.py.")

    # Create model and disable all training operations for speed
    model = JetReconstructionModel(options)
    model.load_state_dict(checkpoint)
    model = model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if cuda:
        model = model.cuda()

    return model


def predict_on_test_dataset(model: JetReconstructionModel, cuda: bool = False):
    full_masks = []
    full_targets = []
    full_predictions = []
    full_classifications = []

    for source_data, *targets in tqdm(model.test_dataloader(), desc="Evaluating Model"):
        if cuda:
            source_data = [x.cuda() for x in source_data]

        predictions, classifications = model.predict_jets_and_particles(*source_data)

        full_targets.append([x[0].numpy() for x in targets])
        full_masks.append([x[1].numpy() for x in targets])
        full_predictions.append([x for x in predictions])
        full_classifications.append([x for x in classifications])

    full_masks = np.concatenate(full_masks, axis=-1)
    full_targets = list(map(np.concatenate, zip(*full_targets)))
    full_predictions = list(map(np.concatenate, zip(*full_predictions)))
    full_classifications = list(map(np.concatenate, zip(*full_classifications)))

    num_jets = model.testing_dataset.source_mask.sum(1).numpy()
    num_jets = num_jets[:full_masks.shape[1]]

    return full_predictions, full_classifications, full_targets, full_masks, num_jets
