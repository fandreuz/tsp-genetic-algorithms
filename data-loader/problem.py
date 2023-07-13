from dataclasses import dataclass
import numpy as np


@dataclass
class Problem:
    name: str
    cost_matrix: np.ndarray
    optimal_tour: np.ndarray

    def __post_init__(self):
        self.cost_matrix = np.asfortranarray(self.cost_matrix)
        self.optimal_tour = np.asfortranarray(self.optimal_tour)

    @property
    def n_nodes(self) -> int:
        return self.cost_matrix.shape[0]
