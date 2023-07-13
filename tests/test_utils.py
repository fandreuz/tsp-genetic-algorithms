import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from tsp_genetic import utils

import numpy as np


def test_inverse_array():
    input = np.array((2, 4, 3, 1), order="F")
    output = utils.inverse_array(input)
    assert (output == (4, 1, 3, 2)).all()


def test_inverse_array2():
    input = np.array((1, 4, 2, 3, 6, 5, 7, 8), order="F")
    output = utils.inverse_array(input)
    assert (output == (1, 3, 4, 2, 6, 5, 7, 8)).all()


def test_out_of_bounds():
    assert utils.out_of_bounds(3, 1, 5)
    assert utils.out_of_bounds(3, 6, 5)
    assert not utils.out_of_bounds(3, 3, 5)
    assert not utils.out_of_bounds(3, 4, 5)
    assert not utils.out_of_bounds(3, 5, 5)


def test_wrap():
    assert utils.wrap_to_top(1, 4) == 1
    assert utils.wrap_to_top(2, 4) == 2
    assert utils.wrap_to_top(3, 4) == 3
    assert utils.wrap_to_top(4, 4) == 4
    assert utils.wrap_to_top(5, 4) == 1
    assert utils.wrap_to_top(6, 4) == 2
