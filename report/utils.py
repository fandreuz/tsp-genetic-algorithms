from pathlib import Path
import sys
import numpy as np

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


def setup_matplotlib():
    mpl.rcParams.update(mpl.rcParamsDefault)
    matplotlib.rc("font", size=16)


def get_color(plot):
    return plot[0].get_color()


def plot_std(generations, data, color):
    plt.fill_between(
        generations,
        data.fitness_mean - data.fitness_std,
        data.fitness_mean + data.fitness_std,
        color=color,
        alpha=0.3,
    )


def plot_data(
    generations, labels, best_data, mean_data=None, should_plot_std=True, n_runs=None
):
    setup_matplotlib()

    if mean_data is not None:
        plt.subplot(1, 2, 1)

    colors = [
        get_color(plt.plot(generations, data.fitness_mean, label=label))
        for data, label in zip(best_data, labels)
    ]
    if should_plot_std:
        for data, label, color in zip(best_data, labels, colors):
            plot_std(generations=generations, data=data, color=color)

    if n_runs is None:
        plt.title("Best")
    else:
        plt.title(f"Best (mean of {n_runs} runs)")
    plt.legend()
    plt.xlabel("Generation")
    plt.grid()

    if mean_data is not None:
        plt.subplot(1, 2, 2)
        for data, label, color in zip(mean_data, labels, colors):
            plt.plot(generations, data.fitness_mean, color, label=label)
            if should_plot_std:
                plot_std(generations=generations, data=data, color=color)

        if n_runs is None:
            plt.title("Mean")
        else:
            plt.title(f"Mean of {n_runs} runs")
        plt.legend()
        plt.xlabel("Generation")
        plt.grid()

    plt.show()


def run_optimizations(
    n: int, problem: Problem, configuration: Configuration
) -> DataStorage:
    datas = [DataStorage() for _ in range(n)]
    for data in datas:
        driver(problem, configuration, data)
    best_matrix = np.vstack([data.fitness_best[None] for data in datas])
    mean_matrix = np.vstack([data.fitness_mean[None] for data in datas])

    n_generations = best_matrix.shape[1]

    collective_best = DataStorage()
    for i in range(n_generations):
        collective_best.log_inspection_message(i + 1, best_matrix[:, i], -1)

    collective_mean = DataStorage()
    for i in range(n_generations):
        collective_mean.log_inspection_message(i + 1, mean_matrix[:, i], -1)

    return collective_best, collective_mean
