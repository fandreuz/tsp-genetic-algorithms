import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "tsp-genetic-py/"))
from evolve import driver
from configuration import Configuration, CrossoverStrategy
from crossover import Crossover

sys.path.append(str(Path(__file__).parent.parent / "data-loader/"))
from load import build_problem

from math import floor

parser = argparse.ArgumentParser(prog="tsp-genetic")
parser.add_argument("-d", "--data", type=str, help="TSP data", required=True)
parser.add_argument("-p", "--population", type=int, help="Population size", default=20)
parser.add_argument(
    "-g", "--generations", type=int, help="N. of generations", default=500
)
parser.add_argument(
    "-e",
    "--elite",
    type=float,
    help="Size of the inter-generational elite",
    default=0.1,
)
parser.add_argument(
    "--print-every",
    type=int,
    help="Generations distance between two population inspection messages",
    default=100,
)
parser.add_argument(
    "-m", "--mutation", type=float, help="Mutation probability", default=0.1
)
parser.add_argument(
    "-c",
    "--crossover",
    type=int,
    choices=[e.value for e in Crossover],
    help="Crossover",
    default=Crossover.CX2.value,
)
parser.add_argument(
    "-s",
    "--crossover-strategy",
    type=int,
    choices=[e.value for e in CrossoverStrategy],
    help="Crossover strategy",
    default=CrossoverStrategy.ALL_IN_ORDER.value,
)

args = parser.parse_args()

problem = build_problem(args.data)
print(f"Problem: {problem}")

configuration = Configuration(
    population_size=args.population,
    elite_size=floor(args.elite * args.population),
    n_generations=args.generations,
    mutation_probability=args.mutation,
    print_every=args.print_every,
    crossover_strategy=CrossoverStrategy(args.crossover_strategy),
    crossover=Crossover(args.crossover),
)
print(f"Configuration: {configuration}")

print(driver(problem, configuration)[1:])
