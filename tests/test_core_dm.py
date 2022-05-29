import numpy as np
import pytest

from mpi4py import MPI
from scalapy import core

comm = MPI.COMM_WORLD

rank = comm.rank
size = comm.size

if size != 4:
    raise Exception("Test needs 4 processes.")

test_context = {"gridshape": (2, 2), "block_shape": (3, 3)}


def test_dm_init():
    with core.shape_context(**test_context):
        dm = core.DistributedMatrix([5, 5])

        # Check global shape
        assert dm.global_shape == (5, 5)

        # Check block size
        assert dm.block_shape == test_context["block_shape"]

        # Check local shape
        shapelist = [(3, 3), (3, 2), (2, 3), (2, 2)]
        assert dm.local_shape == shapelist[rank]


def test_dm_load_5x5():
    """Test that a 5x5 DistributedMatrix is loaded correctly"""
    with core.shape_context(**test_context):
        # Generate matrix
        garr = np.arange(25.0).reshape(5, 5, order='F')

        # Manually extract correct sections
        glist = [garr[:3, :3], garr[:3, 3:], garr[3:, :3], garr[3:, 3:]]

        # Load with DistributedMatrix
        dm = core.DistributedMatrix.from_global_array(garr)

        np.testing.assert_equal(dm.local_array, glist[rank])


@pytest.mark.parametrize("gshape,bshape", [
    ((3, 3), (5, 5)),
    ((132, 109), (21, 11)),
    ((5631, 5), (3, 2)),
    ((81, 81), (90, 2)),
])
def test_dm_cycle(gshape, bshape):
    with core.shape_context(**test_context):
        nr, nc = gshape
        arr = np.arange(nr*nc, dtype=np.float64).reshape(nr, nc, order='F')

        dm = core.DistributedMatrix.from_global_array(arr, block_shape=bshape)
        assert (dm.to_global_array() == arr).all()


def test_dm_redistribute():
    """Test redistribution of matrices with different blocking and process grids"""
    with core.shape_context(**test_context):

        # Generate matrix
        garr = np.arange(25.0).reshape(5, 5, order='F')

        # Create DistributedMatrix
        dm3x3 = core.DistributedMatrix.from_global_array(garr, block_shape=[3, 3])
        dm2x2 = core.DistributedMatrix.from_global_array(garr, block_shape=[2, 2])

        rd2x2 = dm3x3.redistribute(block_shape=[2, 2])

        assert (dm2x2.local_array == rd2x2.local_array).all()

        pc2 = core.ProcessContext([4, 1], comm)

        dmpc2 = core.DistributedMatrix.from_global_array(garr, block_shape=[1, 1], context=pc2)
        rdpc2 = dm3x3.redistribute(block_shape=[1, 1], context=pc2)

        assert (dmpc2.local_array == rdpc2.local_array).all()
