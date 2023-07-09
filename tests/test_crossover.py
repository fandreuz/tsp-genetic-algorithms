import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from tsp_genetic import crossover
crossover.cut_size = 3

import numpy as np


def test_pmx():
    parent1 = np.array((1, 2, 5, 6, 4, 3, 8, 7), order="F")
    parent2 = np.array((1, 4, 2, 3, 6, 5, 7, 8), order="F")

    child1, child2 = crossover.partially_mapped_crossover(parent1, parent2, 0.3)

    assert (child1 == (1, 3, 5, 6, 4, 2, 7, 8)).all()
    assert (child2 == (1, 5, 2, 3, 6, 4, 8, 7)).all()
