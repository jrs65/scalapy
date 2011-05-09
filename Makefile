CC=mpicc -Wall

MPICC=mpicc -Wall

all: tests evtest scarray.so

pdtest: pdgemv
	$(MPICC) -o pdtest pdgemv.c

scarray.so: scarray.pyx setup.py bcutil.c
	export CC=mpicc
	export LDSHARED=mpicc
	python setup.py build_ext --inplace

evtest: evtest.c bcutil.c
#	$(MPICC) -o evtest $^ -mkl -lmkl_scalapack_lp64 -lmkl_blacs_openmpi_lp64 -openmp -lpthread
#	$(MPICC) -o evtest $^ -L$(MKLROOT)/lib/intel64 -lmkl_scalapack_lp64 -lmkl_rt -lmkl_blacs_openmpi_lp64 -fopenmp -lpthread
	$(MPICC) -o evtest $^ -lscalapack-openmpi

bcutil.o: bcutil.h

tests = bctest_1d bctest_2d bctest_mmap

tests : $(tests)

$(tests) : % : %.o bcutil.o
	$(CC) -o $@ $^
clean:
	-rm $(tests) *.o evtest
	-rm -rf build/ sarray.so
	-rm *.pyc