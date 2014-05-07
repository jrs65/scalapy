import numpy as np

from mpi4py import MPI
from scalapy import core

comm = MPI.COMM_WORLD

rank = comm.rank
size = comm.size

if size != 4:
    raise Exception("Test needs 4 processes.")

core.initmpi([2, 2], block_shape=[3, 3])


def test_dm_init():
    dm = core.DistributedMatrix([5, 5])

    # Check global shape
    assert dm.global_shape == (5, 5)

    # Check local shape
    shapelist = [(3, 3), (3, 2), (2, 3), (2, 2)]
    assert dm.local_shape == shapelist[rank]


def test_dm_load_5x5():
    ## Test that a 5x5 DistributedMatrix is loaded correctly.

    # Generate matrix
    garr = np.arange(25.0).reshape(5, 5, order='F')

    # Manually extract correct sections
    glist = [garr[:3, :3], garr[:3, 3:], garr[3:, :3], garr[3:, 3:]]

    # Load with DistributedMatrix
    dm = core.DistributedMatrix.from_global_array(garr)

    assert (dm.local_array == glist[rank]).all()


def test_dm_cycle():
    # Iterate over a variety of global shape and blocks shapes, and perform a
    # global from/to cycle to check for errors.
    gsizes = [[3, 3], [132, 109], [5631, 5], [81, 81]]
    bsizes = [[5, 5], [21, 11], [3, 2], [90, 2]]

    for gs, bs in zip(gsizes, bsizes):
        yield dm_from_to_cycle, gs, bs


def dm_from_to_cycle(gshape, bshape):
    nr, nc = gshape
    arr = np.arange(nr*nc, dtype=np.float64).reshape(nr, nc, order='F')

    dm = core.DistributedMatrix.from_global_array(arr, block_shape=bshape)
    assert (dm.to_global_array() == arr).all()
