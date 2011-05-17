
import scarray
import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD

blocksize = [4, 4]

scarray._blocksize = blocksize

scarray.initmpi()

x, y = np.meshgrid(np.arange(10, dtype=np.float64), np.arange(10, dtype=np.float64))

xm = scarray.LocalMatrix.fromarray(x)
ym = scarray.LocalMatrix.fromarray(y)

for i in range(comm.Get_size()):
    comm.Barrier()
    if comm.Get_rank() == i:
        print "Process %i" % i
        print xm.local_matrix
        print
        print ym.local_matrix
        print

xm.to_file()
if comm.Get_rank() == 0:
    x2 = scarray.matrix_pagealign(x, blocksize)
    y2 = scarray.matrix_pagealign(y, blocksize)
    
    x2.reshape(-1, order='A').tofile("x.dat")
    y2.reshape(-1, order='A').tofile("y.dat")

comm.Barrier()

xm2 = scarray.LocalMatrix.fromfile("x.dat", [10, 10])
ym2 = scarray.LocalMatrix.fromfile("y.dat", [10, 10])


for i in range(comm.Get_size()):
    comm.Barrier()
    if comm.Get_rank() == i:
        print "Process %i" % i
        print xm2.local_matrix
        print
        print ym2.local_matrix
        print


xm3 = scarray.LocalMatrix.fromarray(x)

print "xm == ym:", scarray.matrix_equal(xm, ym)

print "xm == xm2:", scarray.matrix_equal(xm, xm2)
print "xm == xm3:", scarray.matrix_equal(xm, xm3)
