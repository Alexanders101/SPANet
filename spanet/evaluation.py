from glob import glob
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from spanet import JetReconstructionModel, Options
from spanet.dataset.types import Evaluation
from spanet.network.jet_reconstruction.jet_reconstruction_network import extract_predictions

from collections import defaultdict


def tree_concatenate(tree):
    output = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            output[key] = tree_concatenate(value)
        else:
            output[key] = np.concatenate(value)

    return output


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


def evaluate_on_test_dataset(model: JetReconstructionModel) -> Evaluation:
    full_assignments = defaultdict(list)
    full_assignment_probabilities = defaultdict(list)
    full_detection_probabilities = defaultdict(list)

    full_classifications = defaultdict(list)
    full_regressions = defaultdict(list)

    for batch in tqdm(model.test_dataloader(), desc="Evaluating Model"):
        sources = [[x[0].to(model.device), x[1].to(model.device)] for x in batch.sources]
        outputs = model.forward(sources)

        assignment_indices = extract_predictions([
            np.nan_to_num(assignment.detach().cpu().numpy(), -np.inf)
            for assignment in outputs.assignments
        ])

        detection_probabilities = np.stack([
            torch.sigmoid(detection).cpu().numpy()
            for detection in outputs.detections
        ])

        classifications = {
            key: torch.softmax(classification, 1).cpu().numpy()
            for key, classification in outputs.classifications.items()
        }

        regressions = {
            key: value.cpu().numpy()
            for key, value in outputs.regressions.items()
        }

        assignment_probabilities = []
        dummy_index = torch.arange(assignment_indices[0].shape[0])
        for assignment_probability, assignment, symmetries in zip(
            outputs.assignments,
            assignment_indices,
            model.event_info.product_symbolic_groups.values()
        ):
            # Get the probability of the best assignment.
            # Have to use explicit function call here to construct index dynamically.
            assignment_probability = assignment_probability.__getitem__((dummy_index, *assignment.T))

            # Convert from log-probability to probability.
            assignment_probability = torch.exp(assignment_probability)

            # Multiply by the symmetry factor to account for equivalent predictions.
            assignment_probability = symmetries.order() * assignment_probability

            # Convert back to cpu and add to database.
            assignment_probabilities.append(assignment_probability.cpu().numpy())

        for i, name in enumerate(model.event_info.product_particles):
            full_assignments[name].append(assignment_indices[i])
            full_assignment_probabilities[name].append(assignment_probabilities[i])
            full_detection_probabilities[name].append(detection_probabilities[i])

        for key, regression in regressions.items():
            full_regressions[key].append(regression)

        for key, classification in classifications.items():
            full_classifications[key].append(classification)

    return Evaluation(
        tree_concatenate(full_assignments),
        tree_concatenate(full_assignment_probabilities),
        tree_concatenate(full_detection_probabilities),
        tree_concatenate(full_regressions),
        tree_concatenate(full_classifications)
    )

