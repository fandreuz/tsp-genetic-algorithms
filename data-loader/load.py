import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from itertools import dropwhile, takewhile

from problem import Problem, toy_problem

data_directory = Path(__file__).parent.parent / "data"


def _read_xml_file(file: str):
    return ET.parse(str(file)).getroot()


def _extract_problem_structure(name: str):
    file = data_directory / (name + ".xml")
    root = _read_xml_file(file)

    graph = root.find("graph")
    n_vertices = len(graph.findall("vertex"))
    cost_matrix = np.empty((n_vertices, n_vertices - 1), dtype=float)
    for idx, vertex in enumerate(graph.findall("vertex")):
        cost_matrix[idx] = np.fromiter(
            map(lambda edge: edge.attrib["cost"], vertex.findall("edge")), dtype=float
        )

    return root.findtext("name"), cost_matrix


def _isdigit(s):
    return s.strip().isdigit()


def _negate(predicate):
    return lambda x: not predicate(x)


def _extract_optimal_tour(name: str) -> np.ndarray:
    file = data_directory / (name + ".opt.tour")
    with open(file, "r") as file_content:
        tour = [
            int(l)
            for l in takewhile(
                _isdigit,
                dropwhile(_negate(_isdigit), file_content),
            )
        ]
        return tour


def build_problem(name: str) -> Problem:
    if name == "toy":
        return toy_problem

    problem_name, cost_matrix = _extract_problem_structure(name)
    optimal_tour = _extract_optimal_tour(name)
    return Problem(
        name=problem_name, cost_matrix=cost_matrix, optimal_tour=optimal_tour
    )
