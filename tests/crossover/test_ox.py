import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from tsp_genetic import ox

import numpy as np

from test_crossover import same_length, not_repeated, random_paths


def callback(parent1, parent2):
    return ox.order_crossover(parent1, parent2, 3 / len(parent1), 5 / len(parent1))


test_same_length = same_length(callback)

test_not_repeated = not_repeated(callback)


def test_ox():
    parent1 = np.array((3, 4, 8, 2, 7, 1, 6, 5), order="F")
    parent2 = np.array((4, 2, 5, 1, 6, 8, 3, 7), order="F")

    child1, child2 = ox.order_crossover(
        parent1, parent2, 3 / len(parent1), 5 / len(parent1)
    )

    assert (child1 == (5, 6, 8, 2, 7, 1, 3, 4)).all()
    assert (child2 == (4, 2, 7, 1, 6, 8, 5, 3)).all()


def test_ox2():
    parent1 = np.array((1, 2, 5, 6, 4, 3, 8, 7), order="F")
    parent2 = np.array((1, 4, 2, 3, 6, 5, 7, 8), order="F")

    child1, child2 = ox.order_crossover(
        parent1, parent2, 2 / len(parent1), 4 / len(parent1)
    )

    assert (child1 == (2, 3, 5, 6, 4, 7, 8, 1)).all()
    assert (child2 == (5, 4, 2, 3, 6, 8, 7, 1)).all()


def test_ox_substring_zero():
    parent1 = np.array((3, 4, 8, 2, 7, 1, 6, 5), order="F")
    parent2 = np.array((4, 2, 5, 1, 6, 8, 3, 7), order="F")

    child1, child2 = ox.order_crossover(
        parent1, parent2, 3 / len(parent1), 3 / len(parent1)
    )

    assert (child1 == (4, 5, 1, 2, 6, 8, 3, 7)).all()
    assert (child2 == (4, 8, 2, 1, 7, 6, 5, 3)).all()


def test_ox_substring_full():
    parent1 = np.array((3, 4, 8, 2, 7, 1, 6, 5), order="F")
    parent2 = np.array((4, 2, 5, 1, 6, 8, 3, 7), order="F")

    child1, child2 = ox.order_crossover(parent1, parent2, 0, 0.99999)

    assert (child1 == parent1).all()
    assert (child2 == parent2).all()


def test_ox_substring_end():
    parent1 = np.array((3, 4, 8, 2, 7, 1, 6, 5), order="F")
    parent2 = np.array((4, 2, 5, 1, 6, 8, 3, 7), order="F")

    child1, child2 = ox.order_crossover(parent1, parent2, 0.9, 0.9)

    assert (child1 == (4, 2, 1, 6, 8, 3, 7, 5)).all()
    assert (child2 == (3, 4, 8, 2, 1, 6, 5, 7)).all()


def test_ox_substring_start():
    parent1 = np.array((3, 4, 8, 2, 7, 1, 6, 5), order="F")
    parent2 = np.array((4, 2, 5, 1, 6, 8, 3, 7), order="F")

    child1, child2 = ox.order_crossover(parent1, parent2, 0, 0)

    assert (child1 == (3, 2, 5, 1, 6, 8, 7, 4)).all()
    assert (child2 == (4, 8, 2, 7, 1, 6, 5, 3)).all()
