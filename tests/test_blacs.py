
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

if size != 4:
    raise Exception("Test needs 4 processes.")


def test_blacs_import():
    from scalapy import blacs


def test_blacs():
    from scalapy import blacs

    ctxt = blacs.sys2blacs_handle(comm)
    blacs.gridinit(ctxt, 2, 2)
    ranklist = [(0, 0), (0, 1), (1, 0), (1, 1)]

    gi = blacs.gridinfo(ctxt)
    gshape = gi[:2]
    gpos = gi[2:]

    assert gshape == (2, 2)
    assert gpos == ranklist[rank]
