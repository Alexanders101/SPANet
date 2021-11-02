from typing import List, Tuple, Mapping, Iterable, Dict
from configparser import ConfigParser
from collections import OrderedDict, namedtuple
from itertools import chain, permutations

from yaml import load as yaml_load

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import numpy as np

from spanet.network.utilities.group_theory import power_set, complete_symbolic_symmetry_group, complete_symmetry_group


class EventInfo:
    def __init__(self,
                 input_types: Mapping[str, str],
                 input_features: Mapping[str, List[Tuple[str, bool, bool]]],
                 event_particles: Tuple[str, ...],
                 event_permutations: str,
                 targets: Mapping[str, Tuple[Tuple[str, ...], str]]):

        self.input_types = input_types
        self.input_names = list(input_types)
        self.input_features = input_features

        self.event_particles = event_particles
        self.event_permutations = event_permutations

        self.targets = targets
        self.target_mapping = self.variable_mapping(self.targets)
        self.event_symmetries = (len(self.event_particles), eval(self.event_permutations, self.target_mapping))

        self.jet_mappings = OrderedDict()
        self.mapped_targets = OrderedDict()
        for target, (jets, jet_permutations) in self.targets.items():
            num_jets = len(jets)
            jet_mapping = self.variable_mapping(jets)
            mapped_permutations = eval(jet_permutations, jet_mapping)

            self.jet_mappings[target] = jet_mapping
            self.mapped_targets[target] = (num_jets, mapped_permutations)

    def normalized_features(self, input_name):
        return np.array([feature[1] for feature in self.input_features[input_name]])

    def log_features(self, input_name):
        return np.array([feature[2] for feature in self.input_features[input_name]])

    @property
    def event_equivalence_classes(self):
        num_particles = self.event_symmetries[0]
        group = self.event_symbolic_group
        sets = map(frozenset, power_set(range(num_particles)))
        return set(frozenset(frozenset(g(x) for x in s) for g in group.elements) for s in sets)

    @property
    def event_symbolic_group(self):
        return complete_symbolic_symmetry_group(*self.event_symmetries)

    @property
    def event_permutation_group(self):
        return complete_symmetry_group(*self.event_symmetries)

    @property
    def event_transpositions(self):
        event_particles, event_permutations = self.event_symmetries
        return set(chain.from_iterable(map(lambda x: permutations(x, r=2), event_permutations)))

    @property
    def target_permutation_groups(self):
        output = []

        for name, (order, symmetries) in self.mapped_targets.items():
            symmetries = [] if symmetries is None else symmetries
            permutation_group = complete_symmetry_group(order, symmetries)
            output.append((name, permutation_group))

        return OrderedDict(output)

    @property
    def target_symbolic_groups(self):
        output = []

        for name, (order, symmetries) in self.mapped_targets.items():
            symmetries = [] if symmetries is None else symmetries
            permutation_group = complete_symbolic_symmetry_group(order, symmetries)
            output.append((name, permutation_group))

        return OrderedDict(output)

    def num_features(self, input_name: str):
        return len(self.input_features[input_name])

    @staticmethod
    def parse_list(list_string: str):
        return tuple(map(str.strip, list_string.strip("][").strip(")(").split(",")))

    @staticmethod
    def variable_mapping(variables: Iterable):
        # noinspection PyTypeChecker
        return OrderedDict(map(reversed, enumerate(variables)))

    @classmethod
    def read_from_ini(cls, filename: str):
        config = ConfigParser()
        config.read(filename)

        if "INPUTS" in config:
            features_types = OrderedDict([
                (key.upper(), val)
                for key, val in config["INPUTS"].items()
            ])
        else:
            features_types = OrderedDict([("SOURCE", "sequential")])

        print(features_types)
        source_features = OrderedDict(
            (
                key,
                [
                    (
                        name,
                        "normalize" in normalize.lower() or "true" in normalize.lower(),
                        "log" in normalize.lower()
                    )
                    for name, normalize in config[key].items()
                ]
            )
            for key in features_types
        )

        event_particles = cls.parse_list(config["EVENT"]["particles"])
        event_permutations = config["EVENT"]["permutations"]

        targets = OrderedDict()
        for key in event_particles:
            target_jets = cls.parse_list(config[key]["jets"])
            target_permutations = config[key]["permutations"]
            targets[key] = (target_jets, target_permutations)

        return cls(features_types, source_features, event_particles, event_permutations, targets)

    @classmethod
    def read_from_yaml(cls, filename: str):
        with open(filename, 'r') as file:
            config = yaml_load(file, Loader)

        # Extract input feature information
        input_types = OrderedDict()
        input_features = OrderedDict()
        FeatureType = namedtuple("feature", ["name", "normalize", "log_scale"])

        for input_name, input_information in config["INPUTS"].items():
            input_types[input_name] = input_information["TYPE"]
            input_features[input_name] = [
                FeatureType(
                    name,
                    "normalize" in normalize.lower() or "true" in normalize.lower(),
                    "log" in normalize.lower()
                )

                for name, normalize in input_information["FEATURES"].items()
            ]

        # Extract event information
        event_particles = tuple(config["EVENT"]["PARTICLES"])
        event_permutations = list(map(tuple, config["EVENT"]["PERMUTATIONS"]))

        # Extract particle information
        particles = OrderedDict()
        for particle in event_particles:
            particle_jets = config[particle]["JETS"]
            particle_permutations = list(map(tuple, config[particle]["PERMUTATIONS"]))

            particle_reconstructions = OrderedDict([
                list(jet.items())[0] if isinstance(jet, dict) else (jet, [])
                for jet in particle_jets
            ])

            particle_jets = tuple(particle_reconstructions.keys())

            particles[particle] = (particle_jets, particle_permutations, particle_reconstructions)

        return cls(input_types, input_features, event_particles, event_permutations, particles)
