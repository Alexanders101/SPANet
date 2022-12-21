from argparse import ArgumentParser
from typing import List

import torch
from torch.nn import functional as F
from torch.utils._pytree import tree_map

import pytorch_lightning as pl

from spanet import JetReconstructionModel
from spanet.dataset.types import Source
from spanet.evaluation import load_model


class WrappedModel(pl.LightningModule):
    def __init__(
            self,
            model: JetReconstructionModel,
            input_log_transform: bool = False,
            output_log_transform: bool = False
    ):
        super(WrappedModel, self).__init__()

        self.model = model
        self.input_log_transform = input_log_transform
        self.output_log_transform = output_log_transform

    def apply_input_log_transform(self, sources):
        new_sources = []
        for (data, mask), name in zip(sources, self.self.model.event_info.input_names):
            data = torch.clone(data)

            for i, log_transformer in enumerate(self.self.model.event_info.log_features(name)):
                if log_transformer:
                    data[:, :, i].log_()

            new_sources.append(Source(data, mask))
        return new_sources

    def forward(self, sources: List[Source]):
        if self.input_log_transform:
            sources = self.apply_input_log_transform(sources)

        outputs = self.model(sources)

        if self.output_log_transform:
            assignments = [assignment for assignment in outputs.assignments]
            detections = [F.logsigmoid(detection) for detection in outputs.detections]
        else:
            assignments = [assignment.exp() for assignment in outputs.assignments]
            detections = [torch.sigmoid(detection) for detection in outputs.detections]

        regressions = [
            outputs.regressions[key]
            for key in self.model.training_dataset.regressions.keys()
        ]

        classifications = [
            outputs.classifications[key]
            for key in self.model.training_dataset.classifications.keys()
        ]

        return *assignments, *detections, *regressions, *classifications


def onnx_specification(model, output_log_transform: bool = False):
    input_names = []
    output_names = []

    dynamic_axes = {}

    for input_name in model.event_info.input_names:
        for input_type in ["data", "mask"]:
            current_input = f"{input_name}_{input_type}"
            input_names.append(current_input)
            dynamic_axes[current_input] = {
                0: 'batch_size',
                1: f'num_{input_name}'
            }

        if output_log_transform:
            output_names.append(f"{input_name}_assignment_log_probability")
        else:
            output_names.append(f"{input_name}_assignment_probability")

    for input_name in model.event_info.input_names:
        if output_log_transform:
            output_names.append(f"{input_name}_detection_log_probability")
        else:
            output_names.append(f"{input_name}_detection_probability")

    for regression in model.training_dataset.regressions.keys():
        output_names.append(regression)

    for classification in model.training_dataset.classifications.keys():
        output_names.append(classification)

    return input_names, output_names, dynamic_axes


def main(
        log_directory: str,
        output_file: str,
        input_log_transform: bool,
        output_log_transform: bool,
        gpu: bool
):
    model = load_model(log_directory, cuda=gpu)

    # Create wrapped model with flat inputs and outputs
    wrapped_model = WrappedModel(model, input_log_transform, output_log_transform)
    wrapped_model.to(model.device)
    wrapped_model.eval()
    for parameter in wrapped_model.parameters():
        parameter.requires_grad_(False)

    input_names, output_names, dynamic_axes = onnx_specification(model, output_log_transform)

    batch = next(iter(model.train_dataloader()))
    sources = batch.sources
    if gpu:
        sources = tree_map(lambda x: x.cuda(), batch.sources)
    sources = tree_map(lambda x: x[:1], sources)

    print("-" * 60)
    print(f"Compiling network to ONNX model: {output_file}")
    print("-" * 60)
    wrapped_model.to_onnx(
        output_file,
        sources,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("log_directory", type=str,
                        help="Pytorch Lightning Log directory containing the checkpoint and options file.")

    parser.add_argument("output_file", type=str,
                        help="Name to output the ONNX model to.")

    parser.add_argument("-g", "--gpu", action="store_true",
                        help="Trace the network on a gpu.")

    parser.add_argument("--input-log-transform", action="store_true",
                        help="Exported model will apply log transformations to input features automatically.")

    parser.add_argument("--output-log-transform", action="store_true",
                        help="Exported model will output log probabilities. This is more numerically stable.")

    arguments = parser.parse_args()
    main(**arguments.__dict__)
