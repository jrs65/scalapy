import numpy as np

from mpi4py import MPI

from scalapy import core
import scalapy.routines as rt


comm = MPI.COMM_WORLD

rank = comm.rank
size = comm.size

if size != 4:
    raise Exception("Test needs 4 processes.")

test_context = {"gridshape": (2, 2), "block_shape": (16, 16)}


def test_copy_d():
    """Test copy of a real double precision distributed matrix"""
    with core.shape_context(**test_context):

        m, n = 354, 231

        a = np.random.standard_normal((m, n)).astype(np.float64)
        a = np.asfortranarray(a)

        distributed_a = core.DistributedMatrix.from_global_array(a, rank=0)

        distributed_b = rt.copy(distributed_a)
        b = distributed_b.to_global_array(rank=0)

        if rank == 0:
            np.testing.assert_equal(a, b)


def test_copy_z():
    """Test copy of a complex distributed matrix"""
    with core.shape_context(**test_context):

        m, n = 379, 432

        a = np.random.standard_normal((m, n)).astype(np.float64) +\
            1j * np.random.standard_normal((m, n)).astype(np.float64)
        a = np.asfortranarray(a)

        distributed_a = core.DistributedMatrix.from_global_array(a, rank=0)

        distributed_b = rt.copy(distributed_a)
        b = distributed_b.to_global_array(rank=0)

        if rank == 0:
            np.testing.assert_equal(a, b)


def test_triu_d():
    """Test upper-triangular copy of a matrix"""
    with core.shape_context(**test_context):

        m, n = 354, 231

        a = np.random.standard_normal((m, n)).astype(np.float64)
        a = np.asfortranarray(a)

        distributed_a = core.DistributedMatrix.from_global_array(a, rank=0)

        distributed_b = rt.triu(distributed_a)
        b = distributed_b.to_global_array(rank=0)

        if rank == 0:
            np.testing.assert_equal(np.triu(a), b)


def test_tril_d():
    """Test lower-triangular copy of a matrix"""
    with core.shape_context(**test_context):

        m, n = 354, 231

        a = np.random.standard_normal((m, n)).astype(np.float64)
        a = np.asfortranarray(a)

        distributed_a = core.DistributedMatrix.from_global_array(a, rank=0)

        distributed_b = rt.tril(distributed_a)
        b = distributed_b.to_global_array(rank=0)

        if rank == 0:
            np.testing.assert_equal(np.tril(a), b)
