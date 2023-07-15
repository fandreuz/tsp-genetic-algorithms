import argparse
from pathlib import Path
import sys
from utils import enum_content

sys.path.append(str(Path(__file__).parent.parent / "tsp-genetic-py/"))
from evolve import driver
from configuration import (
    Configuration,
    CrossoverStrategy,
    CrossoverRetainment,
    NextGenerationPolicy,
)
from crossover import Crossover
from mutation import Mutation

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
    "-n",
    "--next-generation",
    type=int,
    choices=[e.value for e in NextGenerationPolicy],
    help="Next generation policy-- " + enum_content(NextGenerationPolicy),
    default=NextGenerationPolicy.REPLACE_ALL.value,
)
parser.add_argument(
    "--print-every",
    type=int,
    help="Generations distance between two population inspection messages",
    default=100,
)
parser.add_argument(
    "--mutation-operator",
    type=int,
    help="Mutation operator -- " + enum_content(Mutation),
    default=Mutation.TWORS.value,
)
parser.add_argument(
    "-m", "--mutation-probability", type=float, help="Mutation probability", default=0.1
)
parser.add_argument(
    "--mutation-function-degree",
    type=int,
    help="Degree of the function describing the evolution of the mutation probability",
    default=0,
)
parser.add_argument(
    "-c",
    "--crossover",
    type=int,
    choices=[e.value for e in Crossover],
    help="Crossover -- " + enum_content(CrossoverStrategy),
    default=Crossover.CX2.value,
)
parser.add_argument(
    "-s",
    "--crossover-strategy",
    type=int,
    choices=[e.value for e in CrossoverStrategy],
    help="Crossover strategy -- " + enum_content(Crossover),
    default=CrossoverStrategy.ALL_IN_ORDER.value,
)
parser.add_argument(
    "-r",
    "--crossover-retainment",
    type=int,
    choices=[e.value for e in CrossoverRetainment],
    help="Crossover retainment -- " + enum_content(CrossoverRetainment),
    default=CrossoverRetainment.ALL_CHILDREN,
)

args = parser.parse_args()

problem = build_problem(args.data)
print(f"Problem: {problem}")

configuration = Configuration(
    population_size=args.population,
    elite_size=floor(args.elite * args.population),
    n_generations=args.generations,
    mutation_operator=Mutation(args.mutation_operator),
    mutation_probability=args.mutation_probability,
    mutation_function_degree=args.mutation_function_degree,
    print_every=args.print_every,
    crossover_strategy=CrossoverStrategy(args.crossover_strategy),
    crossover=Crossover(args.crossover),
    crossover_retainment=CrossoverRetainment(args.crossover_retainment),
    next_generation_policy=NextGenerationPolicy(args.next_generation),
)
print(f"Configuration: {configuration}")

print(driver(problem, configuration)[1:])
