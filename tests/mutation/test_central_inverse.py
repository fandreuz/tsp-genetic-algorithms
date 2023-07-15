import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from tsp_genetic import central_inverse

from test_mutation import same_length, not_repeated, random_paths


def callback(chromosome):
    return central_inverse.mutate(chromosome, 3)


test_same_length = same_length(callback)

test_not_repeated = not_repeated(callback)


def test_mutation():
    chromosome = np.array((1, 2, 3, 4, 5, 6, 7), order="F")
    output = central_inverse.mutate(chromosome, 4)
    assert (output == (4, 3, 2, 1, 7, 6, 5)).all()


def test_mutation_start():
    chromosome = np.array((1, 2, 3, 4, 5, 6, 7), order="F")
    output = central_inverse.mutate(chromosome, 1)
    assert (output == (1, 7, 6, 5, 4, 3, 2)).all()


def test_mutation_end():
    chromosome = np.array((1, 2, 3, 4, 5, 6, 7), order="F")
    output = central_inverse.mutate(chromosome, 7)
    assert (output == (7, 6, 5, 4, 3, 2, 1)).all()
