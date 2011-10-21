

CC=mpicc
SCLIBS_INTEL_OPENMPI=-lmkl_scalapack_lp64 -lmkl_rt -lmkl_blacs_openmpi_lp64 -fopenmp -liomp5 -lpthread

SCLIBS=$(SCLIBS_INTEL_OPENMPI)

benchmarks: pdgemm_bench


pdgemm_bench: pdgemm_bench.c
	$(CC) -g -o pdgemm_bench pdgemm_bench.c $(SCLIBS)
