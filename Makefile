.DEFAULT_GOAL := compile

OUTPUT=tsp_genetic
SRC=tsp-genetic-f90/utils.f90 \
	tsp-genetic-f90/crossover/pmx.f90 \
	tsp-genetic-f90/crossover/cx.f90 \
	tsp-genetic-f90/crossover/ox.f90 \
	tsp-genetic-f90/crossover/cx2.f90 \
	tsp-genetic-f90/crossover/cx2-original.f90 \
	tsp-genetic-f90/mutation/twors.f90 \
	tsp-genetic-f90/mutation/central_inverse.f90 \
	tsp-genetic-f90/problem.f90

compile:
	python3 -m numpy.f2py \
		--f90flags="-fcheck=all -Wall -Wextra -fimplicit-none -fbacktrace" \
		-c \
		-m $(OUTPUT) \
		$(SRC)
fast:
	python3 -m numpy.f2py \
	 	-c \
		-m $(OUTPUT) \
		$(SRC)