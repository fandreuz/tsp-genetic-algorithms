from pathlib import Path
import sys
import numpy as np
from typing import Tuple
from operator import attrgetter

sys.path.append(str(Path(__file__).parent.parent / "tsp-genetic-py"))
sys.path.append(str(Path(__file__).parent.parent / "run"))
sys.path.append(str(Path(__file__).parent.parent / "data-loader"))

from configuration import Configuration
from problem import Problem
from evolve import driver
from data_storage import DataStorage

import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl


def get_N_simulations():
    return 25


def setup_matplotlib():
    mpl.rcParams.update(mpl.rcParamsDefault)
    matplotlib.rc("font", size=16)


def _get_color(plot):
    return plot[0].get_color()


def plot_std(generations, data, color):
    plt.fill_between(
        generations,
        data.fitness_mean - data.fitness_std,
        data.fitness_mean + data.fitness_std,
        color=color,
        alpha=0.3,
    )


def run_optimizations(
    problem: Problem, configuration: Configuration, mean=False
) -> DataStorage:
    datas = tuple(DataStorage() for _ in range(get_N_simulations()))
    for data in datas:
        driver(problem, configuration, data)

    if mean:
        matrix = np.vstack([data.fitness_mean[None] for data in datas])
    else:
        matrix = np.vstack([data.fitness_best[None] for data in datas])

    collective = DataStorage()
    for i in range(matrix.shape[1]):
        collective.log_inspection_message(matrix[:, i], -1)

    for i in range(len(datas[0].mutation_probability)):
        collective.log_mutations(datas[0].mutations[i])
        collective.log_mutation_probability(datas[0].mutation_probability[i])

    return collective


def run_many_optimizations(
    problem: Problem, configurations: Tuple[Configuration], mean=False
):
    return tuple(
        run_optimizations(
            problem=problem,
            configuration=configuration,
            mean=mean,
        )
        for configuration in configurations
    )


def plot_data(
    generations,
    labels,
    best_data,
    mean_data=None,
    should_plot_std=False,
    n_runs=None,
    data_operator=attrgetter("fitness_mean"),
    title="Best",
):
    setup_matplotlib()

    if mean_data is not None:
        plt.subplot(1, 2, 1)

    colors = [
        _get_color(plt.plot(generations, data_operator(data), label=label))
        for data, label in zip(best_data, labels)
    ]
    if should_plot_std:
        for data, label, color in zip(best_data, labels, colors):
            plot_std(generations=generations, data=data, color=color)

    if n_runs is None:
        plt.title(title)
    else:
        plt.title(f"{title} (mean of {n_runs} runs)")
    plt.legend()
    plt.xlabel("Generation")
    plt.grid()

    if mean_data is not None:
        plt.subplot(1, 2, 2)
        for data, label, color in zip(mean_data, labels, colors):
            plt.plot(generations, data_operator(data), color, label=label)
            if should_plot_std:
                plot_std(generations=generations, data=data, color=color)

        if n_runs is None:
            plt.title("Mean")
        else:
            plt.title(f"Mean of {n_runs} runs")
        plt.legend()
        plt.xlabel("Generation")
        plt.grid()


def plot_big_and_small(generations, data_small, data_big, labels, **kwargs):
    setup_matplotlib()

    plt.figure(figsize=(20, 6))

    plt.subplot(1, 2, 1)
    plot_data(generations, labels, data_small, **kwargs)

    plt.subplot(1, 2, 2)
    plot_data(generations, labels, data_big, **kwargs)

    plt.show()
