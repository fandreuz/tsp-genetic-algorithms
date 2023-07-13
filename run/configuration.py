from dataclasses import dataclass


@dataclass
class Configuration:
    # Population
    population_size: int
    elite_size: int
    n_generations: int

    # Inspection
    print_every: int

    def __post_init__(self):
        if self.population_size <= 0:
            raise ValueError(f"Population size: {self.population_size} > 0")
        if self.elite_size < 0:
            raise ValueError(f"Elite size: {self.elite_size} >= 0")
        if self.elite_size >= self.population_size:
            raise ValueError(f"Elite size: {self.elite_size} < {self.population_size}")
        if self.mating_size % 2 != 0:
            raise ValueError("Number of matings per generation should be even")
        if self.n_generations <= 0:
            raise ValueError(f"Number of generations: {self.n_generations} > 0")

    @property
    def mating_size(self) -> int:
        return self.population_size - self.elite_size
