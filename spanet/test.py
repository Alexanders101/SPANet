from typing import List, Any, Optional, Dict

from sys import stderr, stdout
from collections import defaultdict
from argparse import ArgumentParser

import numpy as np
from numpy.typing import ArrayLike

from spanet.dataset.evaluator import SymmetricEvaluator, EventInfo
from spanet.evaluation import evaluate_on_test_dataset, load_model
from spanet.dataset.types import Evaluation


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


def transpose_columns(columns):
    header = list(columns.keys())
    num_rows = len(columns[header[0]])

    output = [header]
    for row in range(num_rows):
        output.append([columns[col][row] for col in header])

    return output


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


def display_latex_table(results: Dict[str, Any], jet_limits: List[str], clusters: List[str]):
    columns = " ".join("c" * len(clusters))
    print(r"\begin{tabular}{c | c | c c | c " + columns + "}")
    print(r"\hline")
    print(r"\hline")
    print(r"& $N_\mathrm{jets}$ & Event Proportion  & Jet Proportion & Event Purity & ", end="")
    HEADER_PRINTED = False

    event_types = set(map(lambda x: x.split("/")[0], filter(lambda x: "/" in x, next(iter(results.values())))))
    for event_type in sorted(event_types):
        if "0" in event_type:
            continue

        columns = defaultdict(list)
        for jet_limit in jet_limits:
            particle_keys = [key.split("/")[1] for key in results[jet_limit] if
                             event_type in key and "event" not in key]

            columns["Jet Limit"].append(
                jet_limit.replace(">=", "$\\geq$").replace("==", "$=$")
                if jet_limit is not None
                else jet_limit
            )

            columns["Event Proportion"].append(results[jet_limit][f"{event_type}/event_proportion"])
            columns["Jet Proportion"].append(results[jet_limit][f"event_jet_proportion"])
            columns["Event Purity"].append(results[jet_limit][f"{event_type}/event_purity"])
            for particle_key in sorted(particle_keys):
                name = ' '.join(map(str.capitalize, particle_key.split("_")))
                columns[name].append(results[jet_limit][f"{event_type}/{particle_key}"])

        rows = transpose_columns(columns)
        rows = [[formatter(val) for val in row] for row in rows]

        if not HEADER_PRINTED:
            header = " & ".join(rows[0][4:])
            print(header + r"\\")
            HEADER_PRINTED = True

        print(r"\hline")
        for row_number, row in enumerate(rows[1:]):
            event_name = event_type
            if row_number == len(rows) - 2:
                row = [r"\textbf{" + v + "}" for v in row]
                event_name = r"\textbf{" + event_name + "}"

            row_string = " & ".join(row)
            row_string = "&" + row_string + r"\\"
            if row_number == 0:
                row_string = event_name + row_string

            print(row_string)
        print(r"\hline")

    print(r"\hline")
    print(r"\end{tabular}")


def display_table(results: Dict[str, Any], jet_limits: List[str], clusters: List[str]):
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


def evaluate_predictions(predictions: ArrayLike, num_vectors: ArrayLike, targets: ArrayLike, masks: ArrayLike, event_info_file: str, lines: int):
    event_info = EventInfo.read_from_yaml(event_info_file)
    evaluator = SymmetricEvaluator(event_info)

    minimum_jet_count = num_vectors.min()
    jet_limits = [f"== {minimum_jet_count + i}" for i in range(lines)]
    jet_limits.append(f">= {minimum_jet_count + lines}")
    jet_limits.append(None)

    results = {}
    for jet_limit_name in jet_limits:
        limited_predictions = predictions
        limited_targets = targets
        limited_masks = masks

        if jet_limit_name is not None:
            jet_limit = eval("num_vectors {}".format(jet_limit_name))
            limited_predictions = [p[jet_limit] for p in limited_predictions]
            limited_targets = [t[jet_limit] for t in limited_targets]
            limited_masks = [m[jet_limit] for m in limited_masks]

        results[jet_limit_name] = evaluator.full_report_string(limited_predictions, limited_targets, limited_masks)
        results[jet_limit_name]["event_jet_proportion"] = 1.0 if jet_limit_name is None else jet_limit.mean()

    return results, jet_limits, evaluator.clusters


def main(
    log_directory: str,
    test_file: Optional[str],
    event_file: Optional[str],
    batch_size: Optional[int],
    lines: int,
    gpu: bool,
    latex: bool
):
    model = load_model(log_directory, test_file, event_file, batch_size, gpu)
    evaluation = evaluate_on_test_dataset(model)

    # Flatten predictions
    predictions = list(evaluation.assignments.values())

    # Flatten targets and convert to numpy
    targets = [assignment[0].cpu().numpy() for assignment in model.testing_dataset.assignments.values()]
    masks = [assignment[1].cpu().numpy() for assignment in model.testing_dataset.assignments.values()]

    results, jet_limits, clusters = evaluate_predictions(predictions, model.testing_dataset.num_vectors.cpu().numpy(), targets, masks, model.options.event_info_file, lines)
    if latex:
        display_latex_table(results, jet_limits, clusters)
    else:
        display_table(results, jet_limits, clusters)


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

    parser.add_argument("-l", "--lines", type=int, default=2,
                        help="Number of equality lines to print for every event. "
                             "Will group other events into a >= group.")

    parser.add_argument("-g", "--gpu", action="store_true",
                        help="Evaluate network on the gpu.")

    parser.add_argument("-tex", "--latex", action="store_true",
                        help="Output a latex table.")

    arguments = parser.parse_args()
    main(**arguments.__dict__)


