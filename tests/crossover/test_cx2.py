import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from tsp_genetic import cx2

import numpy as np

from test_crossover import same_length, not_repeated, random_paths


def callback(parent1, parent2):
    return cx2.cycle_crossover2(parent1, parent2)


test_same_length = same_length(callback)

test_not_repeated = not_repeated(callback)


def test_cx2_case1():
    parent1 = np.array((3, 4, 8, 2, 7, 1, 6, 5), order="F")
    parent2 = np.array((4, 2, 5, 1, 6, 8, 3, 7), order="F")

    child1, child2 = callback(parent1, parent2)

    assert (child1 == (4, 8, 6, 2, 5, 3, 1, 7)).all()
    assert (child2 == (1, 7, 4, 8, 6, 2, 5, 3)).all()


def test_cx2_case2():
    parent1 = np.asfortranarray(np.arange(1, 9))
    parent2 = np.array((2, 7, 5, 8, 4, 1, 6, 3), order="F")

    child1, child2 = callback(parent1, parent2)

    assert (child1 == (2, 1, 6, 7, 5, 3, 8, 4)).all()
    assert (child2 == (6, 7, 2, 1, 8, 4, 5, 3)).all()
