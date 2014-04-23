
from mpi4py import MPI

from pyscalapack import blacs

comm = MPI.COMM_WORLD

rank = comm.rank

if comm.size != 4:
    raise Exception("Test needs 4 processes.")


ctxt = blacs.sys2blacs_handle(comm)

print "BLACS handle (rank %i) = %i" % (rank, ctxt)

blacs.gridinit(ctxt, 2, 2)

print "BLACS grid info (rank %i):" % rank, blacs.gridinfo(ctxt)
