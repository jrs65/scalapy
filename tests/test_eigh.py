import numpy as np
import scipy.linalg as la

from mpi4py import MPI

from scalapy import core
import scalapy.routines as rt


comm = MPI.COMM_WORLD

rank = comm.rank
size = comm.size

if size != 4:
    raise Exception("Test needs 4 processes.")

test_context = {"gridshape": (2, 2), "block_shape": (3, 3)}

allclose = lambda a, b: np.allclose(a, b, rtol=1e-4, atol=1e-6)


def cmp_evecs(z1, z2):
    # As there is a freedom in the sign/phase convention (which seems to be
    # different by LAPACK and ScaLAPACK), we need to check in a more
    # sophisticated way.
    c12 = np.dot(z1.T.conj(), z2)
    c12 = np.abs(c12) - np.identity(z1.shape[0])
    return (np.abs(c12) < 1e-6).all()


def test_eigh_D():
    with core.shape_context(**test_context):

        ns = 289

        gA = np.random.standard_normal((ns, ns)).astype(np.float64)
        gA = gA + gA.T  # Make symmetric
        gA = np.asfortranarray(gA)

        dA = core.DistributedMatrix.from_global_array(gA, rank=0)

        evalsd, dZd = rt.eigh(dA)
        gZd = dZd.to_global_array(rank=0)

        if rank == 0:
            evalsn, gZn = la.eigh(gA)

            assert allclose(evalsn, evalsd)
            assert cmp_evecs(gZn, gZd)


def test_eigh_Z():
    with core.shape_context(**test_context):

        ns = 272

        gA = np.random.standard_normal((ns, ns)).astype(np.float64)
        gA = gA + 1.0J * np.random.standard_normal((ns, ns)).astype(np.float64)
        gA = gA + gA.T.conj()  # Make Hermitian
        gA = np.asfortranarray(gA)

        dA = core.DistributedMatrix.from_global_array(gA, rank=0)

        evalsd, dZd = rt.eigh(dA)
        gZd = dZd.to_global_array(rank=0)

        if rank == 0:
            evalsn, gZn = la.eigh(gA)

            assert allclose(evalsn, evalsd)
            assert cmp_evecs(gZn, gZd)


def test_eigh_d_generalized():
    with core.shape_context(**test_context):

        ns = 314

        a = np.random.standard_normal((ns, ns)).astype(np.float64)
        a += a.T  # Make symmetric
        a = np.asfortranarray(a)

        b = np.random.standard_normal((ns, ns)).astype(np.float64)
        b += b.T
        b = b @ b
        b = np.asfortranarray(b)

        a_distributed = core.DistributedMatrix.from_global_array(a, rank=0)
        b_distributed = core.DistributedMatrix.from_global_array(b, rank=0)

        vals, vecs_distributed = rt.eigh(a_distributed, b_distributed)
        vecs = vecs_distributed.to_global_array(rank=0)

        if rank == 0:
            np.testing.assert_allclose(a @ vecs - b @ vecs * vals[None, :], 0, err_msg=f"A @ v - val B @ v = 0", atol=1e-5)
            np.testing.assert_allclose(vecs.T @ b @ vecs, np.eye(ns), err_msg=f"v.T @ b @ v = I", atol=1e-8)


def test_eigh_z_generalized():
    with core.shape_context(**test_context):

        ns = 176

        a = np.random.standard_normal((ns, ns)).astype(np.float64) + 1.j * np.random.standard_normal((ns, ns)).astype(np.float64)
        a += a.conj().T
        a = np.asfortranarray(a)

        b = np.random.standard_normal((ns, ns)).astype(np.float64) + 1.j * np.random.standard_normal((ns, ns)).astype(np.float64)
        b += b.conj().T
        b = b @ b
        b = np.asfortranarray(b)

        a_distributed = core.DistributedMatrix.from_global_array(a, rank=0)
        b_distributed = core.DistributedMatrix.from_global_array(b, rank=0)

        vals, vecs_distributed = rt.eigh(a_distributed, b_distributed)
        vecs = vecs_distributed.to_global_array(rank=0)

        if rank == 0:
            np.testing.assert_allclose(a @ vecs - b @ vecs * vals[None, :], 0, err_msg=f"A @ v - val B @ v = 0", atol=1e-5)
            np.testing.assert_allclose(vecs.conj().T @ b @ vecs, np.eye(ns), err_msg=f"v.T @ b @ v = I", atol=1e-8)
