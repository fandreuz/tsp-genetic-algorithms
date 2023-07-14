from enum import Enum
import sys
from types import MappingProxyType
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from tsp_genetic import cx, cx2, cx2_original, ox, pmx


class Crossover(Enum):
    CX = 1
    CX2 = 2
    CX2_ORIGINAL = 3
    OX = 4
    PMX = 5


# Crossovers which need 2 random values
crossover_needs_2_rnd = frozenset((Crossover.OX, Crossover.PMX))

crossover_functions = MappingProxyType(
    {
        Crossover.OX: ox.order_crossover,
        Crossover.PMX: pmx.partially_mapped_crossover,
        Crossover.CX: cx.cycle_crossover,
        Crossover.CX2: cx2.cycle_crossover2,
        Crossover.CX2_ORIGINAL: cx2_original.cycle_crossover2,
    }
)
