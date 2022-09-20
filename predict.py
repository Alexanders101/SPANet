from argparse import ArgumentParser
from typing import Optional
from numpy import ndarray as Array

import h5py

from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset
from spanet.evaluation import predict_on_test_dataset, load_model


def create_hdf5_output(output_file: str,
                       dataset: JetReconstructionDataset,
                       full_predictions: Array,
                       full_classifications: Array):
    print(f"Creating output file at: {output_file}")
    with h5py.File(output_file, 'w') as output:
        output.create_dataset(f"source/mask", data=dataset.source_mask)
        for i, (feature_name, _, _) in enumerate(dataset.event_info.input_features):
            output.create_dataset(f"source/{feature_name}", data=dataset.sources[:, :, i])

        for i, (particle_name, (jets, _)) in enumerate(dataset.event_info.assignments.items()):
            output.create_dataset(f"{particle_name}/mask", data=full_classifications[i])
            for k, jet_name in enumerate(jets):
                output.create_dataset(f"{particle_name}/{jet_name}", data=full_predictions[i][:, k])


def main(log_directory: str,
         output_file: str,
         test_file: Optional[str],
         event_file: Optional[str],
         batch_size: Optional[int],
         gpu: bool):
    model = load_model(log_directory, test_file, event_file, batch_size, gpu)

    full_predictions, full_classifications, *_ = predict_on_test_dataset(model, gpu)
    create_hdf5_output(output_file, model.testing_dataset, full_predictions, full_classifications)


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
