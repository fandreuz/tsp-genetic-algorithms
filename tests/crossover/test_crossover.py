import numpy as np
import pytest
from itertools import product

rng = np.random.default_rng(seed=0)
base = np.asfortranarray(np.arange(1, 9))


@pytest.fixture
def random_paths():
    return tuple(rng.permutation(base) for _ in range(10))


def same_length(callback):
    def test_same_length(random_paths):
        for parent1, parent2 in product(random_paths, random_paths):
            child1, child2 = callback(parent1, parent2)
            assert len(child1) == len(parent1)
            assert len(child2) == len(parent1)

    return test_same_length


def not_repeated(callback):
    def test_not_repeated(random_paths):
        for parent1, parent2 in product(random_paths, random_paths):
            child1, child2 = callback(parent1, parent2)
            assert len(set(child1)) == len(parent1)
            assert len(set(child2)) == len(parent1)

    return test_not_repeated
