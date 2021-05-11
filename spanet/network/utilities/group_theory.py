from itertools import chain, combinations, starmap, permutations
from typing import List, Tuple

from sympy.combinatorics import PermutationGroup, Permutation


def power_set(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def transpositions(iterable):
    return set(chain.from_iterable(map(lambda x: permutations(x, r=2), iterable)))


def complete_indices(size: int, indices: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    output = indices.copy()
    missing_indices = set(range(size)) - set(chain.from_iterable(indices))
    for index in missing_indices:
        output.append((index,))

    return output


def symbolic_symmetry_group(permutation_indices: List[Tuple[int, ...]]) -> PermutationGroup:
    generators = []

    for indices in permutation_indices:
        if len(indices) > 1:
            generators.extend(starmap(Permutation, combinations(indices, 2)))
        else:
            generators.append(Permutation(indices[0]))

    return PermutationGroup(*generators)


def explicit_symbolic_symmetry_group(permutation_indices: List[Tuple[int, ...]]) -> PermutationGroup:
    generators = starmap(Permutation, permutation_indices)
    return PermutationGroup(*generators)


def symmetry_group(permutation_indices: List[Tuple[int, ...]]) -> List[List[int]]:
    permutation_group = symbolic_symmetry_group(permutation_indices)
    symmetries = map(lambda x: x.array_form, permutation_group.elements)
    return list(symmetries)


def complete_symbolic_symmetry_group(size: int, permutation_indices: List[Tuple[int, ...]]) -> PermutationGroup:
    permutation_indices = complete_indices(size, permutation_indices)
    return symbolic_symmetry_group(permutation_indices)


def complete_symmetry_group(size: int, permutation_indices: List[Tuple[int, ...]]) -> List[List[int]]:
    permutation_indices = complete_indices(size, permutation_indices)
    return symmetry_group(permutation_indices)