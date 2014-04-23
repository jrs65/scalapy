
from mpi4py import MPI

from pyscalapack import core

comm = MPI.COMM_WORLD

rank = comm.rank
size = comm.size

if size != 4:
    raise Exception("Test needs 4 processes.")


pc = core.ProcessContext([2, 2], comm=comm)

for i in range(size):
    if rank == i:
        print "=== PC info (rank %i) ===" % rank
        print "  Context:", pc.blacs_context
        print "  Grid shape:", pc.grid_shape
        print "  Grid position:", pc.grid_position
        print
    comm.Barrier()

