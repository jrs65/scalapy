import numpy as np

from mpi4py import MPI

from pyscalapack import core

comm = MPI.COMM_WORLD

rank = comm.rank
size = comm.size

if size != 4:
    raise Exception("Test needs 4 processes.")

core.initmpi([2, 2], block_shape=[3, 3])

dm = core.DistributedMatrix([5, 5])

for i in range(size):
    if rank == i:
        print "=== DM info (rank %i) ===" % rank
        print "  gshape:", dm.global_shape
        print "  bshape:", dm.block_shape
        print "  ppos:", dm.context.grid_position
        print "  lshape:", dm.local_shape
        print "  desc:", dm.desc
        print dm.indices()[0]
        print dm.indices()[1]
        print
    comm.Barrier()

garr = np.arange(25.0).reshape(5, 5, order='F')

if rank == 0:
    print garr

dm2 = core.DistributedMatrix.from_global_array(garr)

for i in range(size):
    if rank == i:
        print "=== DM arr (rank %i) ===" % rank
        print dm2.local_array
        print
    comm.Barrier()

g2 = dm2.to_global_array(rank=0)

for i in range(size):
    if rank == i:
        print "=== Global arr (rank %i) ===" % rank
        print g2
        print
    comm.Barrier()


g3 = dm2.to_global_array()

for i in range(size):
    if rank == i:
        print "=== Global arr (rank %i) ===" % rank
        print g3
        print
    comm.Barrier()
