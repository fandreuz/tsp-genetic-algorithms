.DEFAULT_GOAL := compile

OUTPUT=tsp_genetic
SRC=tsp-genetic/utils.f90 \
	tsp-genetic/crossover/pmx.f90 \
	tsp-genetic/crossover/cx.f90 \
	tsp-genetic/crossover/ox.f90 \
	tsp-genetic/crossover/cx2.f90 \
	tsp-genetic/crossover/cx2-original.f90 \
	tsp-genetic/mutation/swap.f90

compile:
	python3 -m numpy.f2py \
		--f90flags="-fcheck=all -Wall -Wextra -fimplicit-none -fbacktrace" \
		-c -m \
		$(OUTPUT) $(SRC)
fast:
	python3 -m numpy.f2py \
	 	-c -m \
		$(OUTPUT) $(SRC)