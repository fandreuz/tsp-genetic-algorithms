import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from tsp_genetic import cx

import numpy as np

from test_crossover import same_length, not_repeated, random_paths


def callback(parent1, parent2):
    return cx.cycle_crossover(parent1, parent2)


test_same_length = same_length(callback)

test_not_repeated = not_repeated(callback)


def test_cx():
    parent1 = np.array((1, 2, 3, 4, 5, 6, 7, 8), order="F")
    parent2 = np.array((8, 5, 2, 1, 3, 6, 4, 7), order="F")

    child1, child2 = callback(parent1, parent2)

    assert (child1 == (1, 5, 2, 4, 3, 6, 7, 8)).all()
    assert (child2 == (8, 2, 3, 1, 5, 6, 4, 7)).all()


def test_cx_same():
    parent1 = np.array((3, 4, 8, 2, 7, 1, 6, 5), order="F")
    parent2 = np.array((4, 2, 5, 1, 6, 8, 3, 7), order="F")

    child1, child2 = callback(parent1, parent2)

    assert (child1 == parent1).all()
    assert (child2 == parent2).all()
