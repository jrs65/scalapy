
import scarray
import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD

Br = 4
Bc = 4

scarray.initmpi()

x, y = np.meshgrid(np.arange(10, dtype=np.float64), np.arange(10, dtype=np.float64))

xm = scarray.LocalMatrix.fromarray(x, Br, Bc)
ym = scarray.LocalMatrix.fromarray(y, Br, Bc)

for i in range(comm.Get_size()):
    comm.Barrier()
    if comm.Get_rank() == i:
        print "Process %i" % i
        print xm.local_matrix
        print
        print ym.local_matrix
        print

if comm.Get_rank() == 0:
    x2 = scarray.matrix_pagealign(x, [Br, Bc])
    y2 = scarray.matrix_pagealign(y, [Br, Bc])
    
    x2.reshape(-1, order='A').tofile("x.dat")
    y2.reshape(-1, order='A').tofile("y.dat")

comm.Barrier()

xm2 = scarray.LocalMatrix.fromfile("x.dat", 10, 10, Br, Bc)
ym2 = scarray.LocalMatrix.fromfile("y.dat", 10, 10, Br, Bc)


for i in range(comm.Get_size()):
    comm.Barrier()
    if comm.Get_rank() == i:
        print "Process %i" % i
        print xm2.local_matrix
        print
        print ym2.local_matrix
        print
