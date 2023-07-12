import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

from problem import Problem

data_directory = Path(__file__).parent.parent / "data"


def _read_xml_file(file: str):
    return ET.parse(str(file)).getroot()


def _build_problem_from_root(root: ET.Element) -> Problem:
    name = root.findtext("name")

    graph = root.find("graph")
    n_vertices = len(graph.findall("vertex"))
    cost_matrix = np.empty((n_vertices, n_vertices - 1), dtype=float)
    for idx, vertex in enumerate(graph.findall("vertex")):
        cost_matrix[idx] = np.fromiter(
            map(lambda edge: edge.attrib["cost"], vertex.findall("edge")), dtype=float
        )

    return Problem(name, cost_matrix)


def build_problem(name: str):
    file = data_directory / (name + ".xml")
    root = _read_xml_file(file)
    return _build_problem_from_root(root)
