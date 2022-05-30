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


def test_qr_d_long():
    """Test the QR factorization of a real double precision matrix m < n"""
    with core.shape_context(**test_context):

        a = np.random.standard_normal((357, 478)).astype(np.float64)
        a = np.asfortranarray(a)

        distributed_a = core.DistributedMatrix.from_global_array(a, rank=0)

        distributed_q, distributed_r = rt.qr(distributed_a)
        q = distributed_q.to_global_array(rank=0)
        r = distributed_r.to_global_array(rank=0)

        if rank == 0:
            _q, _r = np.linalg.qr(a)
            np.testing.assert_allclose(q.T @ q, np.eye(q.shape[1]), err_msg="orthonormality", atol=1e-14)
            np.testing.assert_allclose(r, np.triu(r), err_msg="upper-triangularity")
            np.testing.assert_allclose(q @ r, a, err_msg="Q @ R = A")


def test_qr_z_tall():
    """Test the QR factorization of a complex precision matrix m > n"""
    with core.shape_context(**test_context):
        shape = 478, 357

        a = np.random.standard_normal(shape).astype(np.float64) +\
            1j * np.random.standard_normal(shape).astype(np.float64)
        a = np.asfortranarray(a)

        distributed_a = core.DistributedMatrix.from_global_array(a, rank=0)

        distributed_q, distributed_r = rt.qr(distributed_a)
        q = distributed_q.to_global_array(rank=0)
        r = distributed_r.to_global_array(rank=0)

        if rank == 0:
            np.testing.assert_allclose(q.conj().T @ q, np.eye(q.shape[1]), err_msg="orthonormality", atol=1e-6)
            np.testing.assert_allclose(r, np.triu(r), err_msg="upper-triangularity")
            np.testing.assert_allclose(q @ r, a, err_msg="Q @ R = A", atol=1e-6)


def test_qr_z_square():
    """Test the QR factorization of a complex precision matrix m = n"""
    with core.shape_context(**test_context):
        shape = 357, 357

        a = np.random.standard_normal(shape).astype(np.float64) +\
            1j * np.random.standard_normal(shape).astype(np.float64)
        a = np.asfortranarray(a)

        distributed_a = core.DistributedMatrix.from_global_array(a, rank=0)

        distributed_q, distributed_r = rt.qr(distributed_a)
        q = distributed_q.to_global_array(rank=0)
        r = distributed_r.to_global_array(rank=0)

        if rank == 0:
            np.testing.assert_allclose(q.conj().T @ q, np.eye(q.shape[1]), err_msg="orthonormality", atol=1e-6)
            np.testing.assert_allclose(r, np.triu(r), err_msg="upper-triangularity")
            np.testing.assert_allclose(q @ r, a, err_msg="Q @ R = A", atol=1e-6)
