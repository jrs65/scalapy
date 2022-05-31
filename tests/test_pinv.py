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

# TODO: tests fail with the default context
test_context = {"gridshape": (2, 2), "block_shape": (3, 3)}

allclose = lambda a, b: np.allclose(a, b, rtol=1e-4, atol=1e-6)


def test_pinv_D():
    """Test pseudo-inverse computation of a real double precision distributed matrix"""
    with core.shape_context(**test_context):

        m, n = 9, 23

        gA = np.random.standard_normal((m, n)).astype(np.float64)
        gA = np.asfortranarray(gA)

        dA = core.DistributedMatrix.from_global_array(gA, rank=0)

        pinvA = rt.pinv(dA)
        gpinvA = pinvA.to_global_array()
        gpinvA = gpinvA[:n, :]

        if rank == 0:
            assert allclose(gA, np.dot(gA, np.dot(gpinvA, gA)))
            assert allclose(gpinvA, np.dot(gpinvA, np.dot(gA, gpinvA)))
            assert allclose(gpinvA, la.pinv(gA)) # compare with scipy result
            if m == n:
                assert allclose(gpinvA, la.inv(gA)) # compare with scipy result


def test_pinv_Z():
    """Test pseudo-inverse computation of a complex double precision distributed matrix"""
    with core.shape_context(**test_context):

        m, n = 7, 9

        gA = np.random.standard_normal((m, n)).astype(np.float64)
        gA = gA + 1.0J * np.random.standard_normal((m, n)).astype(np.float64)
        gA = np.asfortranarray(gA)

        dA = core.DistributedMatrix.from_global_array(gA, rank=0)

        pinvA = rt.pinv(dA)
        gpinvA = pinvA.to_global_array()
        gpinvA = gpinvA[:n, :]

        if rank == 0:
            assert allclose(gA, np.dot(gA, np.dot(gpinvA, gA)))
            assert allclose(gpinvA, np.dot(gpinvA, np.dot(gA, gpinvA)))
            assert allclose(gpinvA, la.pinv(gA)) # compare with scipy result
            if m == n:
                assert allclose(gpinvA, la.inv(gA)) # compare with scipy result


def test_pinv_Z_alt():
    """Test pseudo-inverse computation of a complex double precision distributed matrix"""
    with core.shape_context(**test_context):
        m, n = 7, 4

        gA = np.random.standard_normal((m, n)).astype(np.float64)
        gA = gA + 1.0J * np.random.standard_normal((m, n)).astype(np.float64)
        gA = np.dot(gA, gA.T.conj())
        assert np.linalg.matrix_rank(gA) < gA.shape[0] # no full rank
        gA = np.asfortranarray(gA)

        m, n = gA.shape

        dA = core.DistributedMatrix.from_global_array(gA, rank=0)

        pinvA = rt.pinv(dA)
        gpinvA = pinvA.to_global_array()
        gpinvA = gpinvA[:n, :]

        if rank == 0:
            assert not allclose(gA, np.dot(gA, np.dot(gpinvA, gA)))
            assert not allclose(gpinvA, np.dot(gpinvA, np.dot(gA, gpinvA)))
            spinvA = la.pinv(gA)
            assert allclose(gA, np.dot(gA, np.dot(spinvA, gA)))
            assert allclose(spinvA, np.dot(spinvA, np.dot(gA, spinvA)))
            assert not allclose(gpinvA, la.pinv(gA)) # compare with scipy result
