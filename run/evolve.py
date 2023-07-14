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


def _elitism(
    population: np.ndarray,
    next_population: np.ndarray,
    fitness: np.ndarray,
    configuration: Configuration,
):
    population = population[np.argsort(fitness)]
    next_population[: configuration.elite_size] = population[: configuration.elite_size]


def _mate(
    population: np.ndarray,
    next_population: np.ndarray,
    mating_indexes_choice: np.ndarray,
    configuration: Configuration,
):
    parent_indexes = rnd.choice(mating_indexes_choice, configuration.mating_size)
    for i in range(0, configuration.mating_size - 2, 2):
        parent1 = population[parent_indexes[i]]
        parent2 = population[parent_indexes[i + 1]]
        (
            next_population[configuration.elite_size + i],
            next_population[configuration.elite_size + i + 1],
        ) = cx2.cycle_crossover2(parent1, parent2)

    if configuration.mating_size % 2 == 0:
        parent1 = population[parent_indexes[-2]]
        parent2 = population[parent_indexes[-1]]
        (
            next_population[-1],
            next_population[-2],
        ) = cx2.cycle_crossover2(parent1, parent2)
    else:
        parent1 = population[parent_indexes[-1]]
        parent2 = population[parent_indexes[0]]
        next_population[-1], _ = cx2.cycle_crossover2(parent1, parent2)


def _mutate(
    next_population: np.ndarray,
    configuration: Configuration,
    mutation_indexes_choice: np.ndarray,
):
    mutations_bitmap = rnd.binomial(
        1, configuration.mutation_probability, configuration.population_size
    ).astype(bool)
    n_mutations = np.sum(mutations_bitmap)

    swap_indices = rnd.choice(mutation_indexes_choice, size=(n_mutations, 2))
    for idx, population_idx in enumerate(np.nonzero(mutations_bitmap)[0]):
        next_population[population_idx] = swap_mutation.swap(
            next_population[population_idx], *swap_indices[idx]
        )

    return n_mutations


def driver(problem: Problem, configuration: Configuration):
    mating_indexes_choice = list(
        range(configuration.elite_size, configuration.population_size)
    )
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

        _elitism(
            population=population,
            next_population=next_population,
            fitness=fitness,
            configuration=configuration,
        )

        _mate(
            population=population,
            next_population=next_population,
            mating_indexes_choice=mating_indexes_choice,
            configuration=configuration,
        )

        mutations_count += _mutate(
            next_population=next_population,
            configuration=configuration,
            mutation_indexes_choice=mutation_indexes_choice,
        )

        population, next_population = next_population, population

    print_mutations(mutations_count)

    fitness = _compute_fitness(problem.cost_matrix, population)
    best = np.argmin(fitness)
    return population[best], fitness[best], optimum
