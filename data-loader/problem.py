from dataclasses import dataclass
import numpy as np


@dataclass
class Problem:
    name: str
    cost_matrix: np.ndarray
    optimal_tour: np.ndarray

    def __post_init__(self):
        assert self.cost_matrix.shape[1] == self.cost_matrix.shape[0] - 1
        assert len(set(self.optimal_tour)) == self.cost_matrix.shape[0]

        self.cost_matrix = np.asfortranarray(self.cost_matrix)
        self.optimal_tour = np.asfortranarray(self.optimal_tour)

    @property
    def n_nodes(self) -> int:
        return self.cost_matrix.shape[0]


_toy_costs = np.zeros((7, 7), dtype=float)
_toy_costs[0, 1:] = [34, 36, 37, 31, 33, 35]
_toy_costs[1, 2:] = [29, 23, 22, 25, 24]
_toy_costs[2, 3:] = [17, 12, 18, 17]
_toy_costs[3, 4:] = [32, 30, 29]
_toy_costs[4, 5:] = [26, 24]
_toy_costs[5, 6:] = [19]
_toy_costs = np.maximum(_toy_costs, _toy_costs.T)
_toy_costs = np.delete(_toy_costs.flatten(), np.arange(0, 50, 8)).reshape(7, 6)

toy_problem = Problem("toy", _toy_costs, np.array([6, 1, 5, 3, 4, 2, 7]))
