from dataclasses import dataclass


@dataclass
class Configuration:
    # Population
    population_size: int
    elite_size: int
    n_generations: int

    # Evolution
    mutation_probability: float

    # Inspection
    print_every: int

    def __post_init__(self):
        if self.population_size <= 0:
            raise ValueError(f"Population size: {self.population_size} > 0")
        if self.elite_size < 0:
            raise ValueError(f"Elite size: {self.elite_size} >= 0")
        if self.elite_size >= self.population_size:
            raise ValueError(f"Elite size: {self.elite_size} < {self.population_size}")
        if self.n_generations <= 0:
            raise ValueError(f"Number of generations: {self.n_generations} > 0")
        if not 0 <= self.mutation_probability <= 1:
            raise ValueError(
                f"Mutation probability: 0 <= {self.mutation_probability} <= 1"
            )

    @property
    def mating_size(self) -> int:
        return self.population_size - self.elite_size
