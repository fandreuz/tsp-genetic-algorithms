import numpy as np
import sys
from pathlib import Path

from configuration import (
    Configuration,
    CrossoverStrategy,
    CrossoverRetainment,
    assign_children_policy,
    NextGenerationPolicy,
)
from crossover import crossover_functions, crossover_needs_2_rnd
from inspection import print_header, print_inspection_message, print_mutations

sys.path.append(str(Path(__file__).parent.parent))
from tsp_genetic import utils, problem as genetic_problem, swap_mutation

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
    idxs = np.argsort(fitness)
    population[:] = population[idxs]
    fitness[:] = fitness[idxs]
    next_population[: configuration.elite_size] = population[: configuration.elite_size]


def _compute_parent_pairs(configuration: Configuration):
    n_children_per_generation = configuration.population_size - configuration.elite_size
    if configuration.crossover_retainment == CrossoverRetainment.ALL_CHILDREN:
        pairs = n_children_per_generation // 2
        if pairs * 2 < n_children_per_generation:
            pairs += 1
        return pairs
    elif configuration.crossover_retainment == CrossoverRetainment.FIRST:
        return n_children_per_generation
    else:
        raise ValueError("Unexpected crossover retainment policy")


def _select_mating_pairs(
    configuration: Configuration,
    mating_indexes_choice: np.ndarray,
    fitness: np.ndarray,
):
    parent_pairs = _compute_parent_pairs(configuration)
    if configuration.crossover_strategy == CrossoverStrategy.ALL_IN_ORDER:
        return np.tile(mating_indexes_choice, 2)[: parent_pairs * 2]
    elif configuration.crossover_strategy == CrossoverStrategy.ALL_RANDOM_PAIRS:
        return rnd.permutation(np.tile(mating_indexes_choice, 2)[: parent_pairs * 2])
    elif configuration.crossover_strategy == CrossoverStrategy.RANDOM_PAIRS:
        return rnd.choice(mating_indexes_choice, parent_pairs * 2)
    elif configuration.crossover_strategy == CrossoverStrategy.FITNESS_RANDOM_PAIRS:
        ifitness = 1 / fitness
        return rnd.choice(
            mating_indexes_choice, parent_pairs * 2, p=ifitness / ifitness.sum()
        )
    else:
        raise ValueError(
            f"Unexpected crossover strategy: {configuration.crossover_strategy}"
        )


def _mate(
    mating_population: np.ndarray,
    next_population: np.ndarray,
    configuration: Configuration,
):
    M = len(mating_population)
    assert M % 2 == 0
    Md2 = M // 2

    if configuration.crossover in crossover_needs_2_rnd:
        additional_args = rnd.random(M).reshape(-1, 2)
    else:
        additional_args = [tuple() for _ in range(Md2)]

    crossover_function = crossover_functions[configuration.crossover]
    assign_children, assign_last_children = assign_children_policy(
        configuration.crossover_retainment
    )

    for i in range(Md2 - 1):
        i2 = i * 2
        p1 = mating_population[i2]
        p2 = mating_population[i2 + 1]
        c1, c2 = crossover_function(p1, p2, *(additional_args[i]))
        assign_children(
            c1=c1,
            c2=c2,
            start_idx=configuration.elite_size,
            parent_pair_idx=i,
            next_population=next_population,
        )

    p1 = mating_population[-2]
    p2 = mating_population[-1]
    c1, c2 = crossover_function(p1, p2, *(additional_args[-1]))
    assign_last_children(
        c1=c1,
        c2=c2,
        start_idx=configuration.elite_size,
        next_population=next_population,
    )


def _mutate(
    next_population: np.ndarray,
    configuration: Configuration,
    mutation_indexes_choice: np.ndarray,
    generation_count: int,
):
    mp = configuration.compute_mutation_probability(generation_count)
    mutations_bitmap = rnd.binomial(1, mp, configuration.population_size).astype(bool)
    n_mutations = np.sum(mutations_bitmap)

    swap_indices = rnd.choice(mutation_indexes_choice, size=(n_mutations, 2))
    for idx, population_idx in enumerate(np.nonzero(mutations_bitmap)[0]):
        next_population[population_idx] = swap_mutation.swap(
            next_population[population_idx], *swap_indices[idx]
        )

    return n_mutations


def driver(problem: Problem, configuration: Configuration):
    mating_indexes_choice = np.arange(
        configuration.elite_size, configuration.population_size
    )
    mutation_indexes_choice = np.arange(1, problem.n_nodes + 1)

    population = _init_population(problem.n_nodes, configuration.population_size)
    next_population = np.empty_like(population, order="F")

    optimum = _compute_fitness(problem.cost_matrix, problem.optimal_tour[None])[0]
    if configuration.print_every > 0:
        print_header()

    mutations_count = 0

    fitness = _compute_fitness(problem.cost_matrix, population)
    for current_generation in range(configuration.n_generations - 1):
        if current_generation % configuration.print_every == 0:
            print_inspection_message(
                current_generation=current_generation,
                fitness=fitness,
                optimum=optimum,
            )

        _elitism(
            configuration=configuration,
            population=population,
            next_population=next_population,
            fitness=fitness,
        )

        mating_pairs = _select_mating_pairs(
            configuration=configuration,
            mating_indexes_choice=mating_indexes_choice,
            fitness=fitness[configuration.elite_size :],
        )
        _mate(
            configuration=configuration,
            mating_population=population[mating_pairs],
            next_population=next_population,
        )

        mutations_count += _mutate(
            generation_count=current_generation,
            configuration=configuration,
            next_population=next_population,
            mutation_indexes_choice=mutation_indexes_choice,
        )
        next_fitness = _compute_fitness(problem.cost_matrix, next_population)

        if configuration.next_generation_policy == NextGenerationPolicy.REPLACE_ALL:
            population, next_population = next_population, population
            fitness = next_fitness
        elif configuration.next_generation_policy == NextGenerationPolicy.BEST:
            tot_fitness = np.concatenate((fitness, next_fitness))
            idxes = np.argsort(tot_fitness)
            keep = idxes[: configuration.population_size]

            fitness = tot_fitness[keep]
            population = np.vstack((population, next_population))[keep]
        else:
            raise ValueError(
                f"Unexpected policy: {configuration.next_generation_policy}"
            )

    current_generation = configuration.n_generations - 1
    if current_generation % configuration.print_every == 0:
        print_inspection_message(
            current_generation=current_generation,
            fitness=fitness,
            optimum=optimum,
        )

    print_mutations(mutations_count)

    best = np.argmin(fitness)
    return population[best], fitness[best], optimum
