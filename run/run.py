import argparse
from pathlib import Path
import sys

from evolve import driver
from configuration import Configuration

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

args = parser.parse_args()

problem = build_problem(args.data)
print(f"Problem: {problem}")

configuration = Configuration(
    population_size=args.population,
    elite_size=floor(args.elite * args.population),
    n_generations=args.generations,
    print_every=args.print_every,
)
print(f"Configuration: {configuration}")

print(driver(problem, configuration)[1:])
