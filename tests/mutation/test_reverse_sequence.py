import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from tsp_genetic import reverse_sequence

from test_mutation import same_length, not_repeated, random_paths


def callback(chromosome):
    return reverse_sequence.mutate(chromosome, 2, 4)


test_same_length = same_length(callback)

test_not_repeated = not_repeated(callback)


def test_mutation():
    chromosome = np.array((1, 2, 3, 4, 5, 6, 7), order="F")
    output = reverse_sequence.mutate(chromosome, 2, 6)
    assert (output == (1, 6, 5, 4, 3, 2, 7)).all()


def test_mutation_same():
    chromosome = np.array((1, 2, 3, 4, 5, 6, 7), order="F")
    output = reverse_sequence.mutate(chromosome, 2, 2)
    assert (output == chromosome).all()


def test_mutation_start():
    chromosome = np.array((1, 2, 3, 4, 5, 6, 7), order="F")
    output = reverse_sequence.mutate(chromosome, 1, 3)
    assert (output == (3, 2, 1, 4, 5, 6, 7)).all()


def test_mutation_end():
    chromosome = np.array((1, 2, 3, 4, 5, 6, 7), order="F")
    output = reverse_sequence.mutate(chromosome, 5, 7)
    assert (output == (1, 2, 3, 4, 7, 6, 5)).all()
