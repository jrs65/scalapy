from __future__ import print_function
import numpy as np

from mpi4py import MPI

from scalapy import core

comm = MPI.COMM_WORLD

rank = comm.rank
size = comm.size

if size != 4:
    raise Exception("Test needs 4 processes.")

test_context = {"gridshape": (2, 2), "block_shape": (3, 3)}

shape = (5, 5)
farr = np.arange(np.prod(shape), dtype=np.float64)
garr = farr.reshape(shape, order='F')

fname = 'IO_test_tmpfile.dat'
fname2 = 'IO_test_tmpfile_write.dat'


def setup_module():
    if rank == 0:
        farr.tofile(fname)


def test_basic_io():
    with core.shape_context(**test_context):
        dm = core.DistributedMatrix.from_global_array(garr)
        dm_read = core.DistributedMatrix.from_file(fname, shape, dtype=np.float64)

        assert (dm.local_array == dm_read.local_array).all()

        dm_read.to_file(fname2)

        if rank == 0:
            farr2 = np.fromfile(fname2, dtype=np.float64)
            print(farr2)
            print(farr)
            assert (farr == farr2).all()


def teardown_module():
    if rank == 0:
        import os
        os.remove(fname)
        os.remove(fname2)