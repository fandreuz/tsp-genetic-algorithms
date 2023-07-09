.DEFAULT_GOAL := compile

compile:
	python3 -m numpy.f2py -c -m tsp_genetic tsp-genetic/utils.f90 tsp-genetic/crossover/pmx.f90