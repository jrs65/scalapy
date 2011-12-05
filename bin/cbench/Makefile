

CC=mpicc
SCLIBS_INTEL_OPENMPI=-lmkl_scalapack_lp64 -lmkl_rt -lmkl_blacs_openmpi_lp64 -fopenmp -liomp5 -lpthread

SCLIBS_INTEL_INTELMPI=-lmkl_scalapack_lp64 -lmkl_rt -lmkl_blacs_intelmpi_lp64 -openmp -liomp5 -lpthread

#SCLIBS=$(SCLIBS_INTEL_INTELMPI)
SCLIBS=$(SCLIBS_INTEL_OPENMPI)

benchmarks: pdgemm_bench dgemm_bench


pdgemm_bench: pdgemm_bench.c
	$(CC) -g -o pdgemm_bench pdgemm_bench.c $(SCLIBS)

dgemm_bench: dgemm_bench.c
	$(CC) -g -o dgemm_bench dgemm_bench.c $(SCLIBS)


clean:
	rm pdgemm_bench dgemm_bench