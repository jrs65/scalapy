import numpy as np

from mpi4py import MPI

from scalapy import core
import scalapy.routines as rt


comm = MPI.COMM_WORLD

rank = comm.rank
size = comm.size

if size != 4:
    raise Exception("Test needs 4 processes.")

core.initmpi([2, 2], block_shape=[16, 16])

allclose = lambda a, b: np.allclose(a, b, rtol=1e-4, atol=1e-6)


def test_trans_D():
    ## Test transpose of a real double precision distributed matrix
    m, n = 354, 231

    gA = np.random.standard_normal((m, n)).astype(np.float64)
    gA = np.asfortranarray(gA)

    dA = core.DistributedMatrix.from_global_array(gA, rank=0)

    dAT = rt.transpose(dA)
    gAT = dAT.to_global_array(rank=0)

    if rank == 0:
        assert allclose(gAT, gA.T) # compare with numpy result


def test_trans_Z():
    ## Test transpose of a complex double precision distributed matrix
    m, n = 379, 432

    gA = np.random.standard_normal((m, n)).astype(np.float64)
    gA = gA + 1.0J * np.random.standard_normal((m, n)).astype(np.float64)
    gA = np.asfortranarray(gA)

    dA = core.DistributedMatrix.from_global_array(gA, rank=0)

    dAT = rt.transpose(dA)
    gAT = dAT.to_global_array(rank=0)

    if rank == 0:
        assert allclose(gAT, gA.T) # compare with numpy result


def test_conj_D():
    ## Test complex conjugate of a real double precision distributed matrix
    m, n = 245, 357

    gA = np.random.standard_normal((m, n)).astype(np.float64)
    gA = np.asfortranarray(gA)

    dA = core.DistributedMatrix.from_global_array(gA, rank=0)

    dAC = rt.conj(dA)
    gAC = dAC.to_global_array(rank=0)

    if rank == 0:
        assert allclose(gAC, gA.conj()) # compare with numpy result


def test_conj_Z():
    ## Test complex conjugate of a complex double precision distributed matrix
    m, n = 630, 62

    gA = np.random.standard_normal((m, n)).astype(np.float64)
    gA = gA + 1.0J * np.random.standard_normal((m, n)).astype(np.float64)
    gA = np.asfortranarray(gA)

    dA = core.DistributedMatrix.from_global_array(gA, rank=0)

    dAC = rt.conj(dA)
    gAC = dAC.to_global_array(rank=0)

    if rank == 0:
        assert allclose(gAC, gA.conj()) # compare with numpy result


def test_hconj_D():
    ## Test Hermitian conjugate of a real double precision distributed matrix
    m, n = 245, 357

    gA = np.random.standard_normal((m, n)).astype(np.float64)
    gA = np.asfortranarray(gA)

    dA = core.DistributedMatrix.from_global_array(gA, rank=0)

    dAH = rt.hconj(dA)
    gAH = dAH.to_global_array(rank=0)

    if rank == 0:
        assert allclose(gAH, gA.T.conj()) # compare with numpy result


def test_hconj_Z():
    ## Test Hermitian conjugate of a complex double precision distributed matrix
    m, n = 630, 62

    gA = np.random.standard_normal((m, n)).astype(np.float64)
    gA = gA + 1.0J * np.random.standard_normal((m, n)).astype(np.float64)
    gA = np.asfortranarray(gA)

    dA = core.DistributedMatrix.from_global_array(gA, rank=0)

    dAH = rt.hconj(dA)
    gAH = dAH.to_global_array(rank=0)

    if rank == 0:
        assert allclose(gAH, gA.T.conj()) # compare with numpy result


if __name__ == '__main__':
    test_trans_D()
    test_trans_Z()
    test_conj_D()
    test_conj_Z()
    test_hconj_D()
    test_hconj_Z()
