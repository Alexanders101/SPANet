from typing import List, Tuple, Mapping, Iterable, Dict, Union
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


def with_default(value, default):
    return default if value is None else value


def key_with_default(database, key, default):
    if key not in database:
        return default

    value = database[key]
    return default if value is None else value


class EventInfo:
    class InputType:
        Global = "GLOBAL"
        Sequential = "SEQUENTIAL"

    class SpecialKey:
        Mask = "MASK"
        Event = "EVENT"
        Inputs = "INPUTS"
        Targets = "TARGETS"
        Particle = "PARTICLE"
        Regressions = "REGRESSIONS"
        Permutations = "PERMUTATIONS"

    Feature = namedtuple("Feature", ["name", "normalize", "log_scale"])

    def __init__(self,
                 input_types: Dict[str, str],
                 input_features: Dict[str, List[Tuple[str, bool, bool]]],
                 event_particles: Tuple[str, ...],
                 event_permutations: Union[str, List[Tuple[str, ...]]],
                 particles: Dict[str, Tuple[Tuple[str, ...], Union[str, List[Tuple[str, ...]]]]],
                 regressions: Dict[str, Union[List[str], Dict[str, List[str]]]]):

        self.input_types = input_types
        self.input_names = list(input_types)
        self.input_features = input_features

        self.sequential_inputs = [
            input_name
            for input_name in self.input_names
            if self.input_types[input_name] == self.InputType.Sequential
        ]

        self.global_inputs = [
            input_name
            for input_name in self.input_names
            if self.input_types[input_name] == self.InputType.Global
        ]

        self.event_particles = event_particles
        self.event_permutations = event_permutations

        self.targets = particles
        self.target_mapping = self.variable_mapping(self.targets)
        self.event_symmetries = (
            len(self.event_particles),
            self.apply_mapping(self.event_permutations, self.target_mapping)
        )

        self.jet_mappings = OrderedDict()
        self.mapped_targets = OrderedDict()
        for target, (jets, jet_permutations) in self.targets.items():
            num_jets = len(jets)
            jet_mapping = self.variable_mapping(jets)
            mapped_permutations = self.apply_mapping(jet_permutations, jet_mapping)

            self.jet_mappings[target] = jet_mapping
            self.mapped_targets[target] = (num_jets, mapped_permutations)

        self.regressions = regressions

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

    def input_type(self, input_name: str):
        return self.input_types[input_name].upper()

    @staticmethod
    def parse_list(list_string: str):
        return tuple(map(str.strip, list_string.strip("][").strip(")(").split(",")))

    @staticmethod
    def variable_mapping(variables: Iterable) -> Dict[str, int]:
        # noinspection PyTypeChecker
        return OrderedDict(map(reversed, enumerate(variables)))

    @staticmethod
    def apply_mapping(permutations: Union[str, List[Tuple[str, ...]]], mapping: Dict[str, int]):
        # Old style which parses a raw string UNSAFE!
        if isinstance(permutations, str):
            return eval(permutations, mapping)

        return [
            tuple(mapping[key] for key in permutation)
            for permutation in permutations
        ]

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

        # Extract input feature information.
        # ----------------------------------
        input_types = OrderedDict()
        input_features = OrderedDict()

        for input_type in config[cls.SpecialKey.Inputs]:
            current_inputs = with_default(config[cls.SpecialKey.Inputs][input_type], default={})

            for input_name, input_information in current_inputs.items():
                input_types[input_name] = input_type.upper()
                input_features[input_name] = [
                    cls.Feature(
                        name,
                        "normalize" in normalize.lower() or "true" in normalize.lower(),
                        "log" in normalize.lower()
                    )

                    for name, normalize in input_information.items()
                ]

        # Extract event and permutation information.
        # ------------------------------------------
        event_particles = tuple(config[cls.SpecialKey.Event].keys())
        permutation_config = key_with_default(config, cls.SpecialKey.Permutations, default={})
        event_permutations = key_with_default(permutation_config, cls.SpecialKey.Event, default=[])
        event_permutations = list(map(tuple, event_permutations))

        daughter_particles = OrderedDict()
        for event_particle in event_particles:
            particle_jets = config[cls.SpecialKey.Event][event_particle]

            particle_permutations = key_with_default(permutation_config, event_particle, default=[])
            particle_permutations = list(map(tuple, particle_permutations))

            daughter_particles[event_particle] = (particle_jets, particle_permutations)

        # Extract Regression information.
        # -------------------------------
        regressions = key_with_default(config, cls.SpecialKey.Regressions, default={})
        if cls.SpecialKey.Event not in regressions:
            regressions[cls.SpecialKey.Event] = []

        for particle in event_particles:
            if particle not in regressions:
                regressions[particle] = {}

            if cls.SpecialKey.Particle not in regressions[particle]:
                regressions[particle][cls.SpecialKey.Particle] = []

            for daughter in daughter_particles[particle][0]:
                if daughter not in regressions[particle]:
                    regressions[particle][daughter] = []

        return cls(input_types, input_features, event_particles, event_permutations, daughter_particles, regressions)
