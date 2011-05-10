CC=mpicc -Wall

MPICC=mpicc -Wall

all: tests evtest scarray.so

pdtest: pdgemv
	$(MPICC) -o pdtest pdgemv.c

scarray.so: scarray.pyx setup.py bcutil.c
	python setup.py build_ext --inplace

#SCFLAGS=-lscalapack-openmpi
SCFLAGS=-L$(MKLROOT)/lib/intel64 -lmkl_scalapack_lp64 -lmkl_rt -lmkl_blacs_openmpi_lp64 -fopenmp -liomp5 -lpthread
#SCFLAGS=-mkl -lmkl_scalapack_lp64 -lmkl_blacs_openmpi_lp64 -openmp -lpthread

evtest: evtest.c bcutil.c
	$(MPICC) -o evtest $^ $(SCFLAGS)

bcutil.o: bcutil.h

tests = bctest_1d bctest_2d bctest_mmap

tests : $(tests)

$(tests) : % : %.o bcutil.o
	$(CC) -o $@ $^ $(SCFLAGS)
clean:
	-rm $(tests) *.o evtest
	-rm -rf build/ scarray.so
	-rm *.pyc