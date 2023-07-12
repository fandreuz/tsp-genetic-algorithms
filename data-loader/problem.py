from dataclasses import dataclass
import numpy as np


@dataclass
class Problem:
    name: str
    cost_matrix: np.ndarray
