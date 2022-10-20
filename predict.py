from argparse import ArgumentParser
from typing import Optional
from numpy import ndarray as Array

import h5py

from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset
from spanet.dataset.types import Evaluation, SpecialKey
from spanet.evaluation import evaluate_on_test_dataset, load_model


def create_hdf5_output(
    output_file: str,
    dataset: JetReconstructionDataset,
    evaluation: Evaluation
):
    print(f"Creating output file at: {output_file}")
    with h5py.File(output_file, 'w') as output:
        # Copy over the source features from the input file.
        with h5py.File(dataset.data_file, 'r') as input_dataset:
            for input_name in input_dataset[SpecialKey.Inputs]:
                for feature_name in input_dataset[SpecialKey.Inputs][input_name]:
                    output.create_dataset(
                        f"{SpecialKey.Inputs}/{input_name}/{feature_name}",
                        data=input_dataset[SpecialKey.Inputs][input_name][feature_name]
                    )

        # Construct the assignment structure. Output both the top assignment and associated probabilities.
        for event_particle in dataset.event_info.event_particles:
            for i, product_particle in enumerate(dataset.event_info.product_particles[event_particle]):
                output.create_dataset(
                    f"{SpecialKey.Targets}/{event_particle}/{product_particle}",
                    data=evaluation.assignments[event_particle][:, i]
                )

            output.create_dataset(
                f"{SpecialKey.Targets}/{event_particle}/assignment_probability",
                data=evaluation.assignment_probabilities[event_particle]
            )

            output.create_dataset(
                f"{SpecialKey.Targets}/{event_particle}/detection_probability",
                data=evaluation.detection_probabilities[event_particle]
            )

            output.create_dataset(
                f"{SpecialKey.Targets}/{event_particle}/marginal_probability",
                data=(
                    evaluation.detection_probabilities[event_particle] *
                    evaluation.assignment_probabilities[event_particle]
                )
            )

        # Simply copy over the structure of the regressions and classifications.
        for name, regression in evaluation.regressions.items():
            output.create_dataset(f"{SpecialKey.Regressions}/{name}", data=regression)

        for name, classification in evaluation.classifications.items():
            output.create_dataset(f"{SpecialKey.Classifications}/{name}", data=classification)


def main(log_directory: str,
         output_file: str,
         test_file: Optional[str],
         event_file: Optional[str],
         batch_size: Optional[int],
         gpu: bool):
    model = load_model(log_directory, test_file, event_file, batch_size, gpu)

    evaluation = evaluate_on_test_dataset(model)
    create_hdf5_output(output_file, model.testing_dataset, evaluation)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("log_directory", type=str,
                        help="Pytorch Lightning Log directory containing the checkpoint and options file.")

    parser.add_argument("output_file", type=str,
                        help="The output HDF5 to create with the new predicted jets for each event.")

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
