import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from tsp_genetic import utils

import numpy as np


def test_inverse_array():
    input = np.empty(5, dtype=int, order="F")
    input[0] = 2
    input[1] = 4
    input[2] = 3
    input[3] = 0
    input[4] = 1

    output = utils.inverse_array(input)
    assert (output == (3, 4, 0, 2, 1)).all()


def test_out_of_bounds():
    assert utils.out_of_bounds(3, 1, 5)
    assert utils.out_of_bounds(3, 6, 5)
    assert not utils.out_of_bounds(3, 3, 5)
    assert not utils.out_of_bounds(3, 4, 5)
    assert not utils.out_of_bounds(3, 5, 5)
