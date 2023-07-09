import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import tsp_genetic

import numpy as np


def test_x():
    input = np.empty(5, dtype=int, order="F")
    input[0] = 2
    input[1] = 4
    input[2] = 3
    input[3] = 0
    input[4] = 1

    output = tsp_genetic.utils.inverse_array(input)
    assert (output == (3, 4, 0, 2, 1)).all()