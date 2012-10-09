

CC=mpicc -traceback
SCLIBS_INTEL_OPENMPI=-lmkl_scalapack_lp64 -lmkl_rt -lmkl_blacs_openmpi_lp64 -fopenmp -liomp5 -lpthread

SCLIBS_INTEL_INTELMPI=-lmkl_scalapack_lp64 -lmkl_rt -lmkl_blacs_intelmpi_lp64 -openmp -liomp5 -lpthread

SCLIBS_NETLIB=-lscalapack -lblacs -mkl=parallel -lifcore

#SCLIBS=$(SCLIBS_INTEL_INTELMPI)
SCLIBS=$(SCLIBS_NETLIB)
#SCLIBS=$(SCLIBS_INTEL_OPENMPI)

benchmarks: pdgemm_bench dgemm_bench pdsyevd_bench pdgetrf_bench pzheevd_bench


pdgemm_bench: pdgemm_bench.c
	$(CC) -g -o pdgemm_bench pdgemm_bench.c $(SCLIBS)

dgemm_bench: dgemm_bench.c
	$(CC) -g -o dgemm_bench dgemm_bench.c $(SCLIBS)


pdsyevd_bench: pdsyevd_bench.c
	$(CC) -g -o pdsyevd_bench pdsyevd_bench.c $(SCLIBS)

pzheevd_bench: pzheevd_bench.c
	$(CC) -g -o pzheevd_bench pzheevd_bench.c $(SCLIBS)


pdgetrf_bench: pdgetrf_bench.c
	$(CC) -g -o pdgetrf_bench pdgetrf_bench.c $(SCLIBS)

clean:
	rm pdgemm_bench dgemm_bench pdsyevd_bench pdgetrf_bench pzheevd_bench