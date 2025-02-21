from typing import List, Any, Optional, Dict

import io
import h5py
import numpy as np
from numpy.typing import ArrayLike

import torch

from spanet.options import Options
from spanet.dataset.evaluator import EventInfo
from spanet.evaluation import load_model
from spanet.dataset.types import SpecialKey, Source, Predictions
from spanet.network.jet_reconstruction.jet_reconstruction_network import default_assignment_fn


class SPANetInterface:
    @property
    def event_info_file(self) -> str:
        return f"{self.checkpoint_path}/event.yaml"

    @property
    def options_file(self) -> str:
        return f"{self.checkpoint_path}/options.json"

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.event_info = EventInfo.read_from_yaml(self.event_info_file)
        self.options = Options.load(self.options_file)
        self.device = None

        dummy_file = self.create_dummy_file()

        self.network = load_model(checkpoint_path, overrides={
            "training_file": dummy_file,
            "validation_file": dummy_file,
            "testing_file": dummy_file,
            "event_info_file": self.event_info_file
        }
        )

    def create_dummy_file(self) -> io.BytesIO:
        """ Create a dummy in-memory h5 file from the event info to feed into Spanet's load function. """
        file = io.BytesIO()

        with h5py.File(file, "w") as f:
            inputs = f.create_group(SpecialKey.Inputs)
            targets = f.create_group(SpecialKey.Targets)

            num_targets = sum(map(lambda x: x.degree, self.event_info.product_symmetries.values()))
            batch_size = self.options.batch_size

            for name, features in self.event_info.input_features.items():
                group = inputs.create_group(name)
                for feature in features:
                    group.create_dataset(feature.name, data=np.ones((batch_size, num_targets), dtype=np.float32))

                group.create_dataset(SpecialKey.Mask, data=np.ones((batch_size, num_targets), dtype=bool))

            target_id = 0
            for particle, daughters in self.event_info.product_particles.items():
                group = targets.create_group(particle)

                for daughter in daughters:
                    group.create_dataset(daughter, data=np.zeros((batch_size,), dtype=np.int64) + target_id)
                    target_id += 1

            regressions_group = f.create_group(SpecialKey.Regressions)
            for event_regression in self.event_info.regressions[SpecialKey.Event]:
                group = regressions_group.create_group(SpecialKey.Event)
                group.create_dataset(event_regression.name, data=np.ones((batch_size,), dtype=np.float32))

            for particle, particle_regressions in self.event_info.regressions.items():
                if particle == SpecialKey.Event:
                    continue

                particle_regressions_group = regressions_group.create_group(particle)

                for daughter, daughter_regressions in particle_regressions.items():
                    group = particle_regressions_group.create_group(daughter)

                    for daughter_regression in daughter_regressions:
                        group.create_dataset(daughter_regression.name, data=np.ones((batch_size,), dtype=np.float32))

        return file

    def sources_from_features(self, features) -> List[Source]:
        """ Convert dictionary input features to compact source inputs for the network. """
        sources = []

        for source, source_features in self.event_info.input_features.items():
            # Load the mask from the guaranteed key
            mask = features[source][SpecialKey.Mask]

            # Stack data, applying log transform if necessary
            data = np.stack([
                np.log(features[source][feature.name] + 1) if feature.log_scale else features[source][feature.name]
                for feature in source_features
            ], axis=-1)

            # Mask out data just to make sure we dont have any strange outputs
            data = np.where(mask[..., None], data, 0)

            # Convert to torch tensors
            data = torch.from_numpy(data).to(self.device)
            mask = torch.from_numpy(mask).to(self.device)

            sources.append(Source(data, mask))

        return sources

    def add_names_to_predictions(self, predictions: Predictions) -> Predictions:
        assignments = {}
        detections = {}

        for i, (particle, daughters) in enumerate(self.event_info.product_particles.items()):
            assignments[particle] = {}
            detections[particle] = predictions.detections[i]

            for j, daughter in enumerate(daughters):
                assignments[particle][daughter] = predictions.assignments[i][:, j]

        return predictions._replace(
            assignments=assignments,
            detections=detections
        )

    def to(self, device: Optional[torch.device] = None) -> "SPANetInterface":
        self.device = device
        self.network = self.network.to(device)
        return self

    def __call__(self, features: Dict[str, Dict[str, ArrayLike]], assignment_fn=default_assignment_fn) -> Predictions:
        sources = self.sources_from_features(features)
        predictions = self.network.predict(sources, assignment_fn=assignment_fn)
        return self.add_names_to_predictions(predictions)

    def __repr__(self):
        info = []

        info.append(repr(self.event_info))
        info.append("\n")

        info.append("Network")
        info.append("=" * 80)
        info.append(repr(self.network))

        return "\n".join(info)

    def __str__(self):
        info = []

        info.append(str(self.event_info))
        info.append("\n")

        info.append("Network")
        info.append("=" * 80)
        info.append(str(self.network))

        return "\n".join(info)
