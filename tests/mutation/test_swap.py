import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from tsp_genetic import swap

from test_mutation import same_length, not_repeated, random_paths


def callback(chromosome):
    return swap.swap_positions(chromosome, 1, 2)


test_same_length = same_length(callback)

test_not_repeated = not_repeated(callback)


def test_mutation():
    chromosome = np.array((3, 4, 8, 2, 7, 1, 6, 5), order="F")
    output = callback(chromosome)
    assert (output == (4, 3, 8, 2, 7, 1, 6, 5)).all()
