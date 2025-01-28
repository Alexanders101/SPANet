from itertools import chain, combinations, starmap
from typing import List, Tuple, Union

from sympy.combinatorics import Permutation as SymbolicPermutation

from spanet.dataset.types import (
    Permutation,
    Permutations,
    MappedPermutation,
    MappedPermutations,
    PermutationGroup,
    SymbolicPermutationGroup
)

# Possible types of permutation that the user can input
RawPermutation = Union[
    List[List[str]],  # Explicit
    List[str]  # Complete Group
]


def expand_permutation(permutation: RawPermutation) -> Permutation:
    if isinstance(permutation[0], list):
        return [tuple(p) for p in permutation]
    else:
        return [tuple(p) for p in combinations(permutation, 2)]


def expand_permutations(permutations: List[RawPermutation]) -> Permutations:
    expanded_permutations = []
    for permutation in permutations:
        if isinstance(permutation[0], list):
            expanded_permutations.append([tuple(p) for p in permutation])
        else:
            expanded_permutations.extend([[tuple(p)] for p in combinations(permutation, 2)])
    return expanded_permutations


def power_set(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def complete_indices(degree: int, permutations: MappedPermutations) -> MappedPermutations:
    """ Add missing elements to a permutation group based on the expected degree. """
    output = permutations.copy()
    missing_indices = set(range(degree)) - set(chain.from_iterable(chain.from_iterable(permutations)))
    for index in missing_indices:
        output.append([(index,)])

    return output


def symbolic_symmetry_group(permutations: MappedPermutations) -> SymbolicPermutationGroup:
    generators = []
    for permutation in permutations:
        symbolic_permutation = SymbolicPermutation
        for element in permutation:
            symbolic_permutation = symbolic_permutation(*element)
        generators.append(symbolic_permutation)

    return SymbolicPermutationGroup(*generators)


def symmetry_group(permutations: MappedPermutations) -> PermutationGroup:
    permutation_group = symbolic_symmetry_group(permutations)
    symmetries = map(lambda x: x.array_form, permutation_group.elements)
    return list(symmetries)


def complete_symbolic_symmetry_group(degree: int, permutations: MappedPermutations) -> SymbolicPermutationGroup:
    permutations = complete_indices(degree, permutations)
    return symbolic_symmetry_group(permutations)


def complete_symmetry_group(degree: int, permutations: MappedPermutations) -> PermutationGroup:
    permutations = complete_indices(degree, permutations)
    return symmetry_group(permutations)


def create_group_string(group: SymbolicPermutationGroup, mapping: List[str]) -> str:
    generators = [p.cyclic_form for p in group.generators]
    generators = complete_indices(group.degree, generators)

    group_string = []
    for generator in generators:
        generator_string = []

        for cycle in generator:
            generator_string.append("(" + ",".join(mapping[i] for i in cycle) + ")")
        
        generator_string = "".join(generator_string)
        group_string.append(generator_string)

    group_string = "".join(group_string)

    return group_string