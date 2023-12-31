import argparse
from pathlib import Path
import sys
import pickle
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
from printer import Printer
from data_storage import DataStorage

sys.path.append(str(Path(__file__).parent.parent / "data-loader/"))
from load import build_problem

from math import floor

parser = argparse.ArgumentParser(prog="tsp-genetic")
parser.add_argument("-d", "--data", type=str, help="TSP data", required=True)
parser.add_argument(
    "--file", type=str, help="File name where to store data", required=False
)
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
    "--mutation-function-adaptive",
    help="Flag to enable adaptive mutation probability function",
    action="store_true",
)
parser.add_argument(
    "--mutation-function-oscillating-amplitude",
    type=float,
    help="Amplitude of the oscillating term",
    default=0,
)
parser.add_argument(
    "--mutation-function-oscillating-cycles",
    type=int,
    help="Number of cycles of the oscillating term during the run",
    default=10,
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
    crossover_operator=Crossover(args.crossover),
    crossover_strategy=CrossoverStrategy(args.crossover_strategy),
    crossover_retainment=CrossoverRetainment(args.crossover_retainment),
    next_generation_policy=NextGenerationPolicy(args.next_generation),
    mutation_operator=Mutation(args.mutation_operator),
    mutation_probability=args.mutation_probability,
    mutation_function_degree=args.mutation_function_degree,
    mutation_function_adaptive=args.mutation_function_adaptive,
    mutation_function_oscillating_amplitude=args.mutation_function_oscillating_amplitude,
    mutation_function_oscillating_cycles=args.mutation_function_oscillating_cycles,
    print_every=args.print_every,
)
print(f"Configuration: {configuration}")

logger = DataStorage() if args.file is not None else Printer()
output = driver(problem, configuration, logger)

if args.file is not None:
    with open(args.file, "wb") as f:
        pickle.dump(logger, f)

print(output[1:])
