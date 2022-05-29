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

test_context = {"gridshape": (2, 2), "block_shape": (16, 16)}

allclose = lambda a, b: np.allclose(a, b, rtol=1e-4, atol=1e-6)



def test_svd_D():
    """Test SVD computation of a real double precision distributed matrix"""
    with core.shape_context(**test_context):

        ms, ns = 235, 326

        gA = np.random.standard_normal((ms, ns)).astype(np.float64)
        gA = np.asfortranarray(gA)

        dA = core.DistributedMatrix.from_global_array(gA, rank=0)

        U, s, VT = rt.svd(dA)
        gU = U.to_global_array(rank=0)
        gVT = VT.to_global_array(rank=0)

        if rank == 0:
            S = np.diag(s)
            assert allclose(gA, np.dot(gU, np.dot(S, gVT)))

            # compare with numpy results
            nU, ns, nVT = np.linalg.svd(gA, full_matrices=False)
            assert allclose(s, ns)
            # this should be careful as there is a freedom in determining the left/right singular vector: for A = U1 * S * V1^H = U2 * S * V2^H, one can proof that U2^H * U1 = (V1^H * V2)^H = D, where D is and diagonal matrix
            assert allclose(np.dot(nU.T.conj(), gU), np.dot(gVT, nVT.T.conj()).T.conj())


def test_svd_Z():
    """Test SVD computation of a complex double precision distributed matrix"""
    with core.shape_context(**test_context):

        ms, ns = 457, 26

        gA = np.random.standard_normal((ms, ns)).astype(np.float64)
        gA = gA + 1.0J * np.random.standard_normal((ms, ns)).astype(np.float64)
        gA = np.asfortranarray(gA)

        dA = core.DistributedMatrix.from_global_array(gA, rank=0)

        U, s, VT = rt.svd(dA)
        gU = U.to_global_array(rank=0)
        gVT = VT.to_global_array(rank=0)

        if rank == 0:
            S = np.diag(s)
            assert allclose(gA, np.dot(gU, np.dot(S, gVT)))

            # compare with numpy results
            nU, ns, nVT = np.linalg.svd(gA, full_matrices=False)
            assert allclose(s, ns)
            assert allclose(np.dot(nU.T.conj(), gU), np.dot(gVT, nVT.T.conj()).T.conj())
