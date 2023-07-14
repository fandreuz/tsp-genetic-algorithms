import numpy as np
import sys
from pathlib import Path

from configuration import Configuration
from inspection import print_header, print_inspection_message, print_mutations

sys.path.append(str(Path(__file__).parent.parent))
from tsp_genetic import utils, problem as genetic_problem, cx2, swap_mutation

sys.path.append(str(Path(__file__).parent.parent / "data-loader/"))
from problem import Problem


rnd = np.random.default_rng()


def _init_population(n_nodes: int, population_size: int) -> np.ndarray:
    population = np.array([utils.scramble(n_nodes) for _ in range(population_size)])
    return np.asfortranarray(population)


def _compute_fitness(cost_matrix: np.ndarray, population: np.ndarray) -> np.ndarray:
    return genetic_problem.fitness(cost_matrix, np.atleast_2d(population))


def driver(problem: Problem, configuration: Configuration):
    mating_indexes_choice = list(range(configuration.mating_size))
    mutation_indexes_choice = list(range(1, problem.n_nodes + 1))

    population = _init_population(problem.n_nodes, configuration.population_size)
    next_population = np.empty_like(population, order="F")

    optimum = _compute_fitness(problem.cost_matrix, problem.optimal_tour[None])[0]
    if configuration.print_every > 0:
        print_header()

    mutations_count = 0

    for current_generation in range(configuration.n_generations - 1):
        fitness = _compute_fitness(problem.cost_matrix, population)

        if current_generation % configuration.print_every == 0:
            print_inspection_message(
                current_generation=current_generation,
                fitness=fitness,
                optimum=optimum,
            )

        ranks = np.argsort(fitness)
        next_population[: configuration.elite_size] = population[
            ranks[: configuration.elite_size]
        ]

        parent_indexes = rnd.choice(mating_indexes_choice, configuration.mating_size)
        for i in range(0, configuration.mating_size, 2):
            parent1 = population[ranks[parent_indexes[i]]]
            parent2 = population[ranks[parent_indexes[i + 1]]]

            (
                next_population[configuration.elite_size + i],
                next_population[configuration.elite_size + i + 1],
            ) = cx2.cycle_crossover2(parent1, parent2)

        mutations_bitmap = rnd.binomial(
            1, configuration.mutation_probability, configuration.population_size
        ).astype(bool)
        n_mutations = np.sum(mutations_bitmap)
        mutations_count += n_mutations

        swap_indices = rnd.choice(mutation_indexes_choice, size=(n_mutations, 2))
        for idx, population_idx in enumerate(np.nonzero(mutations_bitmap)[0]):
            next_population[population_idx] = swap_mutation.swap(
                next_population[population_idx], *swap_indices[idx]
            )

        population, next_population = next_population, population

    print_mutations(mutations_count)

    fitness = _compute_fitness(problem.cost_matrix, population)
    best = np.argmin(fitness)
    return population[best], fitness[best], optimum