import numpy as np

_generation_pad = 12
_stats_pad = 10


def _print_no_newline(s: str):
    print(s, end="")


def _ljust(val, cnt):
    return str(val).ljust(cnt)


def _separator():
    _print_no_newline(" | ")


def print_inspection_message(
    current_generation: int, fitness: np.ndarray, optimum: float
):
    _print_no_newline(_ljust(current_generation, cnt=_generation_pad))
    _separator()
    _print_no_newline(_ljust(np.min(fitness), _stats_pad))
    _separator()
    _print_no_newline(_ljust(np.max(fitness), _stats_pad))
    _separator()
    _print_no_newline(_ljust(np.mean(fitness), _stats_pad))
    _separator()
    print(_ljust(optimum / np.min(fitness), _stats_pad))


def print_header():
    _print_no_newline(f"{'Generation'.ljust(_generation_pad)}")
    _separator()
    _print_no_newline(f"{'Min'.ljust(_stats_pad)}")
    _separator()
    _print_no_newline(f"{'Max'.ljust(_stats_pad)}")
    _separator()
    _print_no_newline(f"{'Avg'.ljust(_stats_pad)}")
    _separator()
    print(f"{'Optimum/Min'.ljust(_stats_pad)}")


def print_mutations(mutations_count):
    print(f"N. of mutations: {mutations_count}")
