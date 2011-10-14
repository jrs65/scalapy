
import pyscalapack as pysc
import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD

blocksize = [4, 4]

pysc.initmpi(blocksize = blocksize)

x, y = np.meshgrid(np.arange(10, dtype=np.float32), np.arange(10, dtype=np.float32))

xm = pysc.DistributedMatrix.fromarray(x)
ym = pysc.DistributedMatrix.fromarray(y)

for i in range(comm.Get_size()):
    comm.Barrier()

    if comm.Get_rank() == i:
        print "Process %i" % i
        print xm.local_array
        print
        print ym.local_array
        print

xm.tofile("x.dat")
ym.tofile("y.dat")

if comm.Get_rank() == 0:
    x2 = pysc.matrix_pagealign(x, blocksize)
    y2 = pysc.matrix_pagealign(y, blocksize)
    
    x2.reshape(-1, order='A').tofile("xa.dat")
    y2.reshape(-1, order='A').tofile("ya.dat")

comm.Barrier()

xm2 = pysc.DistributedMatrix.fromfile("x.dat", [10, 10], np.float32)
ym2 = pysc.DistributedMatrix.fromfile("y.dat", [10, 10], np.float32)

xm3 = pysc.DistributedMatrix.fromfile("xa.dat", [10, 10], np.float32)
ym3 = pysc.DistributedMatrix.fromfile("ya.dat", [10, 10], np.float32)

for i in range(comm.Get_size()):
    comm.Barrier()
    if comm.Get_rank() == i:
        print "Process %i" % i
        print xm2.local_array
        print
        print ym2.local_array
        print


xm4 = pysc.DistributedMatrix.fromarray(x)

print "xm == ym:", pysc.matrix_equal(xm, ym)
print "xm == xm2:", pysc.matrix_equal(xm, xm2)
print "xm == xm3:", pysc.matrix_equal(xm, xm3)
print "xm == xm4:", pysc.matrix_equal(xm, xm4)
