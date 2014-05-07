
from mpi4py import MPI
from scalapy import core

comm = MPI.COMM_WORLD

rank = comm.rank
size = comm.size

poslist = [(0, 0), (0, 1), (1, 0), (1, 1)]

if size != 4:
    raise Exception("Test needs 4 processes.")


def test_process_context():
    pc = core.ProcessContext([2, 2], comm=comm)

    # Test grid shape is correct
    assert pc.grid_shape == (2, 2)

    # Test we have the correct positions
    assert pc.grid_position == poslist[rank]

    # Test the MPI communicator is correct
    assert comm == pc.mpi_comm


def test_initmpi():

    core.initmpi([2, 2], block_shape=[5, 5])

    # Test grid shape is correct
    assert core._context.grid_shape == (2, 2)

    # Test we have the correct positions
    assert core._context.grid_position == poslist[rank]

    # Test the blockshape is set correctly
    assert core._block_shape == (5, 5)
