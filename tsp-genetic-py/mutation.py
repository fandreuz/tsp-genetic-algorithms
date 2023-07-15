from enum import Enum
import sys
from types import MappingProxyType
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from tsp_genetic import twors, central_inverse, reverse_sequence


class Mutation(Enum):
    TWORS = 1
    CENTRAL_INVERSE = 2
    REVERSE_SEQUENCE = 3


mutation_needs_2_rnd = frozenset(
    (
        Mutation.TWORS,
        Mutation.REVERSE_SEQUENCE,
    )
)


mutation_functions = MappingProxyType(
    {
        Mutation.TWORS: twors.mutate,
        Mutation.CENTRAL_INVERSE: central_inverse.mutate,
        Mutation.REVERSE_SEQUENCE: reverse_sequence.mutate,
    }
)
