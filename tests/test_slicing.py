import numpy as np

from mpi4py import MPI
from scalapy import core

comm = MPI.COMM_WORLD

rank = comm.rank
size = comm.size

if size != 4:
    raise Exception("Test needs 4 processes.")

test_context = {"gridshape": (2, 2), "block_shape": (3, 3)}


def test_dm_slicing():
    """Test redistribution of matrices with different blocking and process grids"""
    with core.shape_context(**test_context):
        # Generate matrix
        garr = np.arange(25.0).reshape(5, 5, order='F')

        # Create DistributedMatrix
        dm = core.DistributedMatrix.from_global_array(garr, block_shape=[2, 2])

        # Slice the distributed matrix
        sm = dm[1:4, -3:]

        # Generate a DistributedMatrix from the sliced global matrix
        gslice = garr[1:4, -3:]
        sm2 = core.DistributedMatrix.from_global_array(gslice, block_shape=[2, 2])

        assert (sm.local_array == sm2.local_array).all()
