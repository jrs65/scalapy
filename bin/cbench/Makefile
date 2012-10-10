

CC=mpicc -traceback
SCLIBS_INTEL_OPENMPI=-lmkl_scalapack_lp64 -lmkl_rt -lmkl_blacs_openmpi_lp64 -fopenmp -liomp5 -lpthread

SCLIBS_INTEL_INTELMPI=-lmkl_scalapack_lp64 -lmkl_rt -lmkl_blacs_intelmpi_lp64 -openmp -liomp5 -lpthread

SCLIBS_NETLIB=-lscalapack -lblacs -mkl=parallel -lifcore

#SCLIBS=$(SCLIBS_INTEL_INTELMPI)
SCLIBS=$(SCLIBS_NETLIB)
#SCLIBS=$(SCLIBS_INTEL_OPENMPI)

benchmark_programs=pdgemm_bench dgemm_bench pdsyevd_bench pdgetrf_bench pzheevd_bench

benchmarks: $(benchmark_programs)

%_bench: %_bench.c
	$(CC) -g -o $@ $< $(SCLIBS)

clean:
	rm $(benchmark_programs)