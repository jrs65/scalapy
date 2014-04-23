import numpy as np

from mpi4py import MPI

from pyscalapack import core
import pyscalapack.lowlevel.pblas as pblas_ll


comm = MPI.COMM_WORLD

rank = comm.rank
size = comm.size

if size != 4:
    raise Exception("Test needs 4 processes.")

core.initmpi([2, 2], block_shape=[3, 3])

dm = core.DistributedMatrix([5, 5])

gA = np.asfortranarray(np.arange(25.0, dtype=np.float64).reshape(5, 5))
gB = np.asfortranarray(np.ones((5, 3), dtype=np.float64))

if rank == 0:

    print '=== gA ==='
    print gA
    print
    print '=== gB ==='
    print gB
    print
    print '=== gC ==='
    print np.dot(gA, gB)
    print

gC = np.zeros((5, 3), dtype=np.float64)



dA = core.DistributedMatrix.from_global_array(gA)
dB = core.DistributedMatrix.from_global_array(gB)
dC = core.DistributedMatrix.from_global_array(gC)

pblas_ll.pdgemm('N', 'N', 5, 3, 5, 1.0,
                np.ravel(dA.local_array, order='A'), 1, 1, dA.desc,
                np.ravel(dB.local_array, order='A'), 1, 1, dB.desc,
                0.0,
                np.ravel(dC.local_array, order='A'), 1, 1, dC.desc)

gC2 = dC.to_global_array(rank=0)

if rank == 0:
    print '=== gC2 ==='
    print gC2
    print
