import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from tsp_genetic import pmx

import numpy as np

from test_crossover import same_length, not_repeated, random_paths


def callback(parent1, parent2):
    return pmx.partially_mapped_crossover(parent1, parent2, 3, 5)


test_same_length = same_length(callback)

test_not_repeated = not_repeated(callback)


def test_pmx():
    parent1 = np.array((1, 2, 5, 6, 4, 3, 8, 7), order="F")
    parent2 = np.array((1, 4, 2, 3, 6, 5, 7, 8), order="F")

    child1, child2 = pmx.partially_mapped_crossover(parent1, parent2, 3, 5)

    assert (child1 == (1, 3, 5, 6, 4, 2, 7, 8)).all()
    assert (child2 == (1, 5, 2, 3, 6, 4, 8, 7)).all()


def test_pmx2():
    parent1 = np.array((3, 4, 8, 2, 7, 1, 6, 5), order="F")
    parent2 = np.array((4, 2, 5, 1, 6, 8, 3, 7), order="F")

    child1, child2 = pmx.partially_mapped_crossover(parent1, parent2, 4, 6)

    assert (child2 == (3, 4, 2, 1, 6, 8, 7, 5)).all()
    assert (child1 == (4, 8, 5, 2, 7, 1, 3, 6)).all()
