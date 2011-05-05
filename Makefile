CC=icc

all: tests pdtest

pdtest: pdgemv
	$(CC) -o pdtest pdgemv.c

evtest: evtest.c bcutil.c
	mpicc -o evtest $^ -mkl -lmkl_scalapack_lp64 -lmkl_blacs_openmpi_lp64 -openmp -lpthread

bcutil.o: bcutil.h

tests = bctest_1d bctest_2d bctest_mmap

tests : $(tests)

$(tests) : % : %.o bcutil.o
	$(CC) -o $@ $^
clean:
	rm $(tests) *.o