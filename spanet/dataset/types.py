from enum import Enum
from copy import deepcopy
from typing import NamedTuple, Dict, Union, TypeVar, List, Tuple, Iterable, Set, FrozenSet, OrderedDict, Callable, Mapping, Optional

import sympy.combinatorics
from numpy.typing import NDArray, ArrayLike, DTypeLike

import numpy as np
from torch import Tensor
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten


ODict = OrderedDict

Permutation = List[Tuple[str, ...]]
MappedPermutation = List[Tuple[int, ...]]

Permutations = List[Permutation]
MappedPermutations = List[MappedPermutation]

PermutationGroup = List[List[int]]
SymbolicPermutationGroup = sympy.combinatorics.PermutationGroup


class Particles:
    names: Tuple[str, ...]
    permutations: Permutations
    sources: Tuple[int, ...]

    def __init__(
        self,
        particles: Tuple[str, ...],
        permutations: Optional[Permutations] = None,
        sources: Optional[Tuple[int, ...]] = None
    ):
        self.names = particles
        self.permutations = permutations if permutations is not None else []
        self.sources = sources if sources is not None else tuple(-1 for _ in self.names)

    def __iter__(self) -> Iterable[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, item) -> str:
        return self.names[item]


# A 1-2 level dictionary structure for storing info about Feynmen diagrams.
Key = TypeVar("Key")
Value = TypeVar("Value")
NewValue = TypeVar("NewValue")
FeynmanDict = Dict[Key, Union[Value, Dict[Key, Value]]]

# To help differentiate that this object can only take event particles as keys.
EventDict = ODict

# To help differentiate that this object can only take product particles as keys.
ProductDict = ODict

# To help differentiate that this object can only take input sources particles as keys.
InputDict = ODict


# Similar to `tree_map` but only recurses over dictionaries so we can have lists of data at each node.
def feynman_map(function: Callable[[Value], NewValue], tree: FeynmanDict[Key, Value]) -> FeynmanDict[Key, NewValue]:
    return {
        key: feynman_map(function, value) if isinstance(value, dict) else function(value)
        for key, value in tree.items()
    }


def feynman_fill(
        tree: FeynmanDict[str, Value],
        event_particles: Particles,
        daughter_particles: Mapping[str, Particles],
        constructor: Callable[[], Value]
):
    tree = deepcopy(tree)

    if SpecialKey.Event not in tree:
        tree[SpecialKey.Event] = constructor()

    for particle in event_particles:
        if particle not in tree:
            tree[particle] = {}

        if SpecialKey.Particle not in tree[particle]:
            tree[particle][SpecialKey.Particle] = constructor()

        for daughter in daughter_particles[particle]:
            if daughter not in tree[particle]:
                tree[particle][daughter] = constructor()

    return tree


class Symmetries(NamedTuple):
    degree: int
    permutations: MappedPermutations


class RegressionInfo(NamedTuple):
    name: str
    type: str = "gaussian"


class FeatureInfo(NamedTuple):
    name: str
    normalize: bool
    log_scale: bool


ClassificationInfo = str


class SpecialKey(str, Enum):
    Mask = "MASK"
    Event = "EVENT"
    Inputs = "INPUTS"
    Targets = "TARGETS"
    Particle = "PARTICLE"
    Regressions = "REGRESSIONS"
    Permutations = "PERMUTATIONS"
    Classifications = "CLASSIFICATIONS"
    Embeddings = "EMBEDDINGS"


class Source(NamedTuple):
    data: Tensor
    mask: Tensor


class Statistics(NamedTuple):
    location: Tensor
    scale: Tensor


class InputType(str, Enum):
    Global = "GLOBAL"
    Relative = "RELATIVE"
    Sequential = "SEQUENTIAL"


class AssignmentTargets(NamedTuple):
    indices: Tensor
    mask: Tensor


class Batch(NamedTuple):
    sources: Tuple[Source, ...]
    num_vectors: int
    assignment_targets: Tuple[AssignmentTargets, ...]
    regression_targets: Dict[str, Tensor]
    classification_targets: Dict[str, Tensor]


class Outputs(NamedTuple):
    assignments: List[Tensor]
    detections: List[Tensor]
    vectors: Dict[str, Tensor]
    regressions: Dict[str, Tensor]
    classifications: Dict[str, Tensor]


class Predictions(NamedTuple):
    assignments: List[NDArray[np.int64]]
    detections: NDArray[np.float32]
    regressions: Dict[str, NDArray[np.float32]]
    classifications: Dict[str, NDArray[np.int64]]


class Evaluation(NamedTuple):
    assignments: Dict[str, NDArray[np.int64]]
    assignment_probabilities: Dict[str, NDArray[np.float32]]
    detection_probabilities: Dict[str, NDArray[np.float32]]
    regressions: Dict[str, NDArray[np.float32]]
    classifications: Dict[str, NDArray[np.float32]]


