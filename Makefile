CC=icc

MPICC=mpicc

all: tests evtest scarray.so

pdtest: pdgemv
	$(MPICC) -o pdtest pdgemv.c

scarray.so: scarray.pyx setup.py bcutil.c
	python setup.py build_ext --inplace

evtest: evtest.c bcutil.c
	$(MPICC) -o evtest $^ -mkl -lmkl_scalapack_lp64 -lmkl_blacs_openmpi_lp64 -openmp -lpthread

bcutil.o: bcutil.h

tests = bctest_1d bctest_2d bctest_mmap

tests : $(tests)

$(tests) : % : %.o bcutil.o
	$(CC) -o $@ $^
clean:
	rm $(tests) *.o