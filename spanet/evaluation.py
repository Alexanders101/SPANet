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
    full_num_jets = []
    full_detections = []
    full_assignments = []

    for batch in tqdm(model.test_dataloader(), desc="Evaluating Model"):
        sources, num_jets, targets, regression_targets, classification_targets = batch
        sources = tuple((x[0].to(model.device), x[1].to(model.device)) for x in sources)
        assignments, detections = model.predict_assignments_and_detections(sources)

        full_targets.append([x[0].numpy() for x in targets])
        full_masks.append([x[1].numpy() for x in targets])
        full_assignments.append([x for x in assignments])
        full_detections.append([x for x in detections])
        full_num_jets.append(num_jets)

    full_num_jets = np.concatenate(full_num_jets)
    full_masks = np.concatenate(full_masks, axis=-1)
    full_targets = list(map(np.concatenate, zip(*full_targets)))
    full_assignments = list(map(np.concatenate, zip(*full_assignments)))
    full_detections = list(map(np.concatenate, zip(*full_detections)))

    return full_assignments, full_detections, full_targets, full_masks, full_num_jets
