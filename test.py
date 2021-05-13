from typing import List, Any, Optional, Dict

from sys import stderr, stdout
from collections import defaultdict
from argparse import ArgumentParser

import numpy as np

from spanet import JetReconstructionModel
from spanet.dataset.evaluator import SymmetricEvaluator, EventInfo
from spanet.evaluation import predict_on_test_dataset, load_model


def formatter(value: Any) -> str:
    """ A monolithic formatter function to convert possible values to output strings.

    """
    if isinstance(value, str):
        return value

    if value is None:
        return "Full"

    if np.isnan(value):
        return "N/A"

    return "{:.3f}".format(value)


# Table function taken from here:
# https://stackoverflow.com/questions/5909873/how-can-i-pretty-print-ascii-tables-with-python
def create_table(table: dict, full_row: bool = False, event_type: str = None) -> None:
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
    header_full_line = ''
    key_list = list(table.keys())
    rc_len_values = []
    for key in key_list:
        rc_len = len(max((v for v in table[key]), key=lambda q: len(str(q))))
        rc_len_values += ([rc_len, [key]] for n in range(len(table[key])))

        heading_line = (key + (" " * (rc_len + (additional_spacing + 1)))) + heading_separator
        rc_separator += ("-" * (len(key) + (rc_len + (additional_spacing + 1)))) + '+-'
        header_full_line += heading_line

    stdout.flush()
    if event_type is not None:
        stdout.write('\n' + rc_separator + '\n' + "Event Type: " + event_type + '\n')
    stdout.write(rc_separator + '\n' + header_full_line + '\n' + rc_separator + '\n')

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


def evaluate_model(model: JetReconstructionModel, cuda: bool = False):
    predictions, _, targets, masks, num_jets = predict_on_test_dataset(model, cuda)

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
        results[jet_limit_name]["event_jet_proportion"] = 1.0 if jet_limit_name is None else jet_limit.mean()

    return results, jet_limits


def display_table(results: Dict[str, Any], jet_limits: List[str]):
    event_types = set(map(lambda x: x.split("/")[0], filter(lambda x: "/" in x, next(iter(results.values())))))
    for event_type in sorted(event_types):
        columns = defaultdict(list)
        for jet_limit in jet_limits:
            particle_keys = [key.split("/")[1] for key in results[jet_limit] if
                             event_type in key and "event" not in key]

            columns["Jet Limit"].append(jet_limit)
            columns["Event Proportion"].append(results[jet_limit][f"{event_type}/event_proportion"])
            columns["Jet Proportion"].append(results[jet_limit][f"event_jet_proportion"])
            columns["Event Purity"].append(results[jet_limit][f"{event_type}/event_purity"])
            for particle_key in sorted(particle_keys):
                name = ' '.join(map(str.capitalize, particle_key.split("_")))
                columns[name].append(results[jet_limit][f"{event_type}/{particle_key}"])

        create_table(columns, event_type=event_type)
        print()


def main(log_directory: str,
         test_file: Optional[str],
         event_file: Optional[str],
         batch_size: Optional[int],
         gpu: bool):
    model = load_model(log_directory, test_file, event_file, batch_size, gpu)
    results, jet_limits = evaluate_model(model, gpu)
    display_table(results, jet_limits)


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

    parser.add_argument("-g", "--gpu", action="store_true",
                        help="Evaluate network on the gpu.")

    arguments = parser.parse_args()
    main(**arguments.__dict__)


