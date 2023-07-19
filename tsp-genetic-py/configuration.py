from dataclasses import dataclass
from enum import Enum
from crossover import Crossover
from mutation import Mutation
import numpy as np
from math import floor


class CrossoverStrategy(Enum):
    # All parents are mated in fitness order
    ALL_IN_ORDER = 1
    # All parents are mated in random pairs
    ALL_RANDOM_PAIRS = 2
    # Random pairs of parents
    RANDOM_PAIRS = 3
    # Random pairs of parents with probability depending on fitness
    FITNESS_RANDOM_PAIRS = 4


class CrossoverRetainment(Enum):
    # All children generated during a mating are retained
    ALL_CHILDREN = 1
    # Only first child is retained
    FIRST = 2


def assign_children_policy(crossover_retainment: CrossoverRetainment):
    if crossover_retainment == CrossoverRetainment.ALL_CHILDREN:

        def _assign_children(
            c1: np.ndarray,
            c2: np.ndarray,
            start_idx: int,
            parent_pair_idx: int,
            next_population: np.ndarray,
        ):
            (
                next_population[start_idx + parent_pair_idx * 2],
                next_population[start_idx + parent_pair_idx * 2 + 1],
            ) = (
                c1,
                c2,
            )

        def _assign_last_children(
            c1: np.ndarray,
            c2: np.ndarray,
            start_idx: int,
            next_population: np.ndarray,
        ):
            if (next_population.shape[0] - start_idx) % 2 == 0:
                next_population[-2], next_population[-1] = c1, c2
            else:
                next_population[-1] = c1

        return _assign_children, _assign_last_children

    elif crossover_retainment == CrossoverRetainment.FIRST:

        def _assign_children(
            c1: np.ndarray,
            c2: np.ndarray,
            start_idx: int,
            parent_pair_idx: int,
            next_population: np.ndarray,
        ):
            next_population[start_idx + parent_pair_idx] = c1

        def _assign_last_children(
            c1: np.ndarray,
            c2: np.ndarray,
            start_idx: int,
            next_population: np.ndarray,
        ):
            next_population[-1] = c1

    else:
        raise ValueError("Unexpected crossover retainment policy")

    return _assign_children, _assign_last_children


class NextGenerationPolicy(Enum):
    # Next generation will replace entirely the old one
    REPLACE_ALL = 1
    # The best chromosomes are retained according to fitness
    BEST = 2


@dataclass
class Configuration:
    # Population
    population_size: int
    elite_size: int
    n_generations: int

    mutation_operator: Mutation
    mutation_probability: float
    mutation_function_degree: int
    mutation_function_adaptive: bool

    crossover_operator: Crossover
    crossover_strategy: CrossoverStrategy
    crossover_retainment: CrossoverRetainment
    next_generation_policy: NextGenerationPolicy

    # Inspection
    print_every: int

    def __post_init__(self):
        if self.population_size <= 0:
            raise ValueError(f"Population size: {self.population_size} > 0")
        if self.elite_size < 0:
            raise ValueError(f"Elite size: {self.elite_size} >= 0")
        if self.elite_size < 1:
            self.elite_size = floor(self.elite_size * self.population_size)
        if self.elite_size >= self.population_size:
            raise ValueError(f"Elite size: {self.elite_size} < {self.population_size}")
        if self.n_generations <= 0:
            raise ValueError(f"Number of generations: {self.n_generations} > 0")
        if not 0 <= self.mutation_probability <= 1:
            raise ValueError(
                f"Mutation probability: 0 <= {self.mutation_probability} <= 1"
            )
        if self.mutation_function_degree < 0:
            raise ValueError(
                f"Mutation function degree: {self.mutation_function_degree} >= 0"
            )
        if self.mutation_function_adaptive and self.mutation_function_degree != 0:
            raise ValueError(
                "mutation_function_adaptive=True is incompatible with non-zero mutation_function_degree"
            )
        if (
            self.next_generation_policy == NextGenerationPolicy.BEST
            and self.elite_size > 0
        ):
            raise ValueError(
                "elite_size > 0 is incompatible with NextGenerationPolicy.BEST"
            )

        if self.mutation_function_degree > 0:
            mutation_function_parameter = (
                self.mutation_probability
                / (self.n_generations - 1) ** self.mutation_function_degree
            )

            def compute_mutation_probability(generation_count):
                return (
                    generation_count**self.mutation_function_degree
                    * mutation_function_parameter
                )

        else:

            def compute_mutation_probability(generation_count):
                return self.mutation_probability

        self.compute_mutation_probability = compute_mutation_probability

        if self.print_every <= 0:
            # Never
            self.print_every = self.n_generations + 1
