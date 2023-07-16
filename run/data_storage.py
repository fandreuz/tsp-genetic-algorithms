import numpy as np
import sys
from pathlib import Path

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

sys.path.append(str(Path(__file__).parent.parent / "tsp-genetic-py/"))
from idata_logger import IDataLogger


class DataStorage(IDataLogger):
    def __init__(self):
        self._mutations = np.array([])
        self._fitness_best = np.array([])
        self._fitness_worst = np.array([])
        self._fitness_mean = np.array([])
        self._fitness_std = np.array([])

    @property
    def mutations(self):
        return self._mutations

    @property
    def fitness_best(self):
        return self._fitness_best

    @property
    def fitness_worst(self):
        return self._fitness_worst

    @property
    def fitness_mean(self):
        return self._fitness_mean

    @property
    def fitness_std(self):
        return self._fitness_std

    @override
    def log_header(self):
        pass

    @override
    def log_inspection_message(
        self, current_generation: int, fitness: np.ndarray, optimum: float
    ):
        self._fitness_best = np.concatenate((self._fitness_best, (np.min(fitness),)))
        self._fitness_worst = np.concatenate((self._fitness_worst, (np.max(fitness),)))
        self._fitness_mean = np.concatenate((self._fitness_mean, (np.mean(fitness),)))
        self._fitness_std = np.concatenate(
            (self._fitness_std, (np.std(fitness, ddof=1),))
        )

    @override
    def log_mutations(self, mutations_count):
        self._mutations = mutations_count
