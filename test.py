from typing import List, Any, Optional, Dict

from sys import stderr, stdout
from collections import defaultdict
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import numpy as np
import torch

from spanet import JetReconstructionModel, Options
from spanet.dataset.evaluator import SymmetricEvaluator, EventInfo


def formatter(value: Any) -> str:
    if isinstance(value, str):
        return value

    if value is None:
        return "Full"

    if np.isnan(value):
        return "N/A"

    return "{:.3f}".format(value)


# Table function taken from here:
# https://stackoverflow.com/questions/5909873/how-can-i-pretty-print-ascii-tables-with-python
def create_table(table: dict, full_row: bool = False) -> None:
    table = {
        k: list(map(formatter, v)) for k, v in table.items()
    }

    min_len = len(min((v for v in table.values()), key=lambda q: len(q)))
    max_len = len(max((v for v in table.values()), key=lambda q: len(q)))

    if min_len < max_len:
        stderr.write("Table is out of shape, please make sure all columns have the same length.")
        stderr.flush()
        return

    additional_spacing = 1

    heading_separator = '| '
    horizontal_split = '| '

    rc_separator = ''
    key_list = list(table.keys())
    rc_len_values = []
    for key in key_list:
        rc_len = len(max((v for v in table[key]), key=lambda q: len(str(q))))
        rc_len_values += ([rc_len, [key]] for n in range(len(table[key])))

        heading_line = (key + (" " * (rc_len + (additional_spacing + 1)))) + heading_separator
        stdout.write(heading_line)

        rc_separator += ("-" * (len(key) + (rc_len + (additional_spacing + 1)))) + '+-'

        if key is key_list[-1]:
            stdout.flush()
            stdout.write('\n' + rc_separator + '\n')

    value_list = [v for vl in table.values() for v in vl]

    aligned_data_offset = max_len

    row_count = len(key_list)

    next_idx = 0
    newline_indicator = 0
    iterations = 0

    for n in range(len(value_list)):
        key = rc_len_values[next_idx][1][0]
        rc_len = rc_len_values[next_idx][0]

        line = ('{:{}} ' + " " * len(key)).format(value_list[next_idx],
                                                  str(rc_len + additional_spacing)) + horizontal_split

        if next_idx >= (len(value_list) - aligned_data_offset):
            next_idx = iterations + 1
            iterations += 1
        else:
            next_idx += aligned_data_offset

        if newline_indicator >= row_count:
            if full_row:
                stdout.flush()
                stdout.write('\n' + rc_separator + '\n')
            else:
                stdout.flush()
                stdout.write('\n')

            newline_indicator = 0

        stdout.write(line)
        newline_indicator += 1

    stdout.write('\n' + rc_separator + '\n')
    stdout.flush()


def load_model(log_directory: str,
               testing_file: Optional[str] = None,
               event_info_file: Optional[str] = None,
               batch_size: Optional[int] = None,
               cuda: bool = False):
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
    if options.testing_file is None:
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

    for source_data, *targets in tqdm(model.test_dataloader(), desc="Evaluating Model"):
        if cuda:
            source_data = [x.cuda() for x in source_data]

        predictions = model.predict_jets(*source_data)

        full_targets.append([x[0].numpy() for x in targets])
        full_masks.append([x[1].numpy() for x in targets])
        full_predictions.append([x for x in predictions])

    full_masks = np.concatenate(full_masks, axis=-1)
    full_targets = list(map(np.concatenate, zip(*full_targets)))
    full_predictions = list(map(np.concatenate, zip(*full_predictions)))

    num_jets = model.testing_dataset.source_mask.sum(1).numpy()
    num_jets = num_jets[:full_masks.shape[1]]

    return full_predictions, full_targets, full_masks, num_jets


def evaluate_model(model: JetReconstructionModel, cuda: bool = False):
    predictions, targets, masks, num_jets = predict_on_test_dataset(model, cuda)

    event_info = EventInfo.read_from_ini(model.options.event_info_file)
    evaluator = SymmetricEvaluator(event_info)

    minimum_jet_count = num_jets.min()
    jet_limits = [f"== {minimum_jet_count}",
                  f"== {minimum_jet_count + 1}",
                  f">= {minimum_jet_count + 2}",
                  None]

    results = {}
    for jet_limit_name in jet_limits:
        limited_predictions = predictions
        limited_targets = targets
        limited_masks = masks

        if jet_limit_name is not None:
            jet_limit = eval("num_jets {}".format(jet_limit_name))
            limited_predictions = [p[jet_limit] for p in limited_predictions]
            limited_targets = [t[jet_limit] for t in limited_targets]
            limited_masks = [m[jet_limit] for m in limited_masks]

        results[jet_limit_name] = evaluator.full_report_string(limited_predictions, limited_targets, limited_masks)

    return results, jet_limits


def display_table(results: Dict[str, Any], jet_limits: List[str], table_length: int):
    event_types = set(map(lambda x: x.split("/")[0], next(iter(results.values()))))
    for event_type in sorted(event_types):
        print("=" * table_length)
        print("{}".format(event_type))
        print("=" * table_length)

        columns = defaultdict(list)
        for jet_limit in jet_limits:
            particle_keys = [key.split("/")[1] for key in results[jet_limit] if
                             event_type in key and "event" not in key]

            columns["Jet Limit"].append(jet_limit)
            columns["Event Proportion"].append(results[jet_limit][f"{event_type}/event_proportion"])
            columns["Event Purity"].append(results[jet_limit][f"{event_type}/event_purity"])
            for particle_key in sorted(particle_keys):
                name = ' '.join(map(str.capitalize, particle_key.split("_")))
                columns[name].append(results[jet_limit][f"{event_type}/{particle_key}"])

        create_table(columns)
        print()


def main(log_directory: str,
         test_file: Optional[str],
         event_file: Optional[str],
         batch_size: Optional[int],
         cuda: bool,
         table_length: int):
    model = load_model(log_directory, test_file, event_file, batch_size, cuda)
    results, jet_limits = evaluate_model(model, cuda)
    display_table(results, jet_limits, table_length)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("log_directory", type=str,
                        help="Pytorch Lightning Log directory containing the checkpoint and options file.")

    parser.add_argument("-tf", "--test_file", type=str, default=None,
                        help="Replace the test file in the options with a custom one. "
                             "Must provide if options does not define a test file.")

    parser.add_argument("-ef", "--event_file", type=str, default=None,
                        help="Replace the event file in the options with a custom event.")

    parser.add_argument("-bs", "--batch_size", type=int, default=None,
                        help="Replace the batch size in the options with a custom size.")

    parser.add_argument("-t", "--table_length", type=int, default=100,
                        help="Size of the output table.")

    parser.add_argument("-c", "--cuda", action="store_true",
                        help="Evaluate network on the gpu.")

    arguments = parser.parse_args()
    main(**arguments.__dict__)


