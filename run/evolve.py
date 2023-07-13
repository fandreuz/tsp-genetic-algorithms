import numpy as np
import sys
from pathlib import Path

from configuration import Configuration

sys.path.append(str(Path(__file__).parent.parent))
from tsp_genetic import utils, problem as genetic_problem, cx2

sys.path.append(str(Path(__file__).parent.parent / "data-loader/"))
from problem import Problem


def _init_population(n_nodes: int, population_size: int) -> np.ndarray:
    population = np.array([utils.scramble(n_nodes) for _ in range(population_size)])
    return np.asfortranarray(population)


def _compute_fitness(cost_matrix: np.ndarray, population: np.ndarray) -> np.ndarray:
    return genetic_problem.fitness(cost_matrix, np.atleast_2d(population))


def driver(problem: Problem, configuration: Configuration):
    mating_indexes_choice = list(range(configuration.mating_size))

    population = _init_population(problem.n_nodes, configuration.population_size)
    next_population = np.empty_like(population, order="F")

    for current_generation in range(configuration.n_generations - 1):
        fitness = _compute_fitness(problem.cost_matrix, population)

        ranks = np.argsort(fitness)
        next_population[: configuration.elite_size] = population[
            ranks[: configuration.elite_size]
        ]

        parent_indexes = np.random.choice(
            mating_indexes_choice, configuration.mating_size
        )
        for i in range(0, configuration.mating_size, 2):
            parent1 = population[ranks[parent_indexes[i]]]
            parent2 = population[ranks[parent_indexes[i + 1]]]

            (
                next_population[configuration.elite_size + i],
                next_population[configuration.elite_size + i + 1],
            ) = cx2.cycle_crossover2(parent1, parent2)

        population, next_population = next_population, population

    fitness = _compute_fitness(problem.cost_matrix, population)
    best = np.argmin(fitness)
    optimum = _compute_fitness(problem.cost_matrix, problem.optimal_tour[None])[0]
    return population[best], fitness[best], optimum
