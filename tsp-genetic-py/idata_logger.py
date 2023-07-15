from abc import ABC, abstractmethod
import numpy as np


class IDataLogger(ABC):
    @abstractmethod
    def log_header(self):
        pass

    @abstractmethod
    def log_inspection_message(
        self, current_generation: int, fitness: np.ndarray, optimum: float
    ):
        pass

    @abstractmethod
    def log_mutations(self, mutations_count):
        pass
