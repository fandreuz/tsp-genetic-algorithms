import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from tsp_genetic import problem

import numpy as np


def test_fitness():
    costs = np.arange(12).reshape(4, 3).astype(float)
    # fmt:off
    population = np.array([
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [2, 1, 4, 3],
    ]) 
    expected_fitness = np.array([
        0 + 4 + 8 + 9,
        11 + 7 + 3 + 2,
        3 + 2 + 11 + 7
    ]).astype(float)
    # fmt:on
    assert (problem.fitness(costs, population) == expected_fitness).all()
