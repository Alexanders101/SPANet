from glob import glob
from typing import Optional, Union, Tuple

import numpy as np
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map

from rich import progress

from spanet import JetReconstructionModel, Options
from spanet.dataset.types import Evaluation, Outputs, Source
from spanet.network.jet_reconstruction.jet_reconstruction_network import extract_predictions

from collections import defaultdict


def dict_concatenate(tree):
    output = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            output[key] = dict_concatenate(value)
        else:
            output[key] = np.concatenate(value)

    return output


def tree_concatenate(trees):
    leaves = []
    for tree in trees:
        data, tree_spec = tree_flatten(tree)
        leaves.append(data)

    results = [np.concatenate(l) for l in zip(*leaves)]
    return tree_unflatten(results, tree_spec)


def load_model(
    log_directory: str,
    testing_file: Optional[str] = None,
    event_info_file: Optional[str] = None,
    batch_size: Optional[int] = None,
    cuda: bool = False,
    checkpoint: Optional[str] = None
) -> JetReconstructionModel:
    # Load the best-performing checkpoint on validation data
    if checkpoint is None:
        checkpoint = sorted(glob(f"{log_directory}/checkpoints/epoch*"))[-1]
        print(f"Loading: {checkpoint}")

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

    # Create model and disable all training operations for speed
    model = JetReconstructionModel(options)
    model.load_state_dict(checkpoint)
    model = model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if cuda:
        model = model.cuda()

    return model


def evaluate_on_test_dataset(
        model: JetReconstructionModel,
        progress=progress,
        return_full_output: bool = False
) -> Union[Evaluation, Tuple[Evaluation, Outputs]]:
    full_assignments = defaultdict(list)
    full_assignment_probabilities = defaultdict(list)
    full_detection_probabilities = defaultdict(list)

    full_classifications = defaultdict(list)
    full_regressions = defaultdict(list)

    full_outputs = []

    dataloader = model.test_dataloader()
    if progress:
        dataloader = progress.track(model.test_dataloader(), description="Evaluating Model")

    for batch in dataloader:
        sources = tuple(Source(x[0].to(model.device), x[1].to(model.device)) for x in batch.sources)
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

        if return_full_output:
            full_outputs.append(tree_map(lambda x: x.cpu().numpy(), outputs))

    evaluation = Evaluation(
        dict_concatenate(full_assignments),
        dict_concatenate(full_assignment_probabilities),
        dict_concatenate(full_detection_probabilities),
        dict_concatenate(full_regressions),
        dict_concatenate(full_classifications)
    )

    if return_full_output:
        return evaluation, tree_concatenate(full_outputs)

    return evaluation

