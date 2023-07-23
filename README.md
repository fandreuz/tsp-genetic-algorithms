# tsp-genetic-algorithms
Review of GA solutions for TSP.

### Strategy

The high level part of the project is implemented in Python (e.g. user input, problem loading, optimization loop, data collection), and the low-level subroutines for crossover and mutation operators are implemented in Fortran 90. The binding between the two languages is provided by [f2py](https://numpy.org/doc/stable/f2py/).
