import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "crossover"))
from test_crossover import random_paths


def same_length(callback):
    def test_same_length(random_paths):
        for chromosome in random_paths:
            mutated = callback(chromosome)
            assert len(mutated) == len(chromosome)

    return test_same_length


def not_repeated(callback):
    def test_not_repeated(random_paths):
        for chromosome in random_paths:
            mutated = callback(chromosome)
            assert len(set(mutated)) == len(chromosome)

    return test_not_repeated
