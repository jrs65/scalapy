import numpy as np

from mpi4py import MPI

from scalapy import core
import scalapy.routines as rt


comm = MPI.COMM_WORLD

rank = comm.rank
size = comm.size

if size != 4:
    raise Exception("Test needs 4 processes.")

core.initmpi([2, 2], block_shape=[3, 3])

allclose = lambda a, b: np.allclose(a, b, rtol=1e-4, atol=1e-6)


def test_np2self_D():
    ## Test copy a numpy array to a section of the distributed matrix and vice versa
    am, an = 13, 5
    Am, An = 39, 23
    srow, scol = 3, 12

    a = np.arange(am*an, dtype=np.float64).reshape(am, an)
    a = np.asfortranarray(a)

    gA = np.random.standard_normal((Am, An)).astype(np.float64)
    gA = np.asfortranarray(gA)

    dA = core.DistributedMatrix.from_global_array(gA, rank=0)

    dA = dA.np2self(a, srow, scol, rank=0)
    a1 = dA.self2np(srow, am, scol, an, rank=1)

    if rank == 1:
        assert allclose(a, a1)


def test_np2self_Z():
    ## Test copy a numpy array to a section of the distributed matrix and vice versa
    am, an = 13, 5
    Am, An = 39, 23
    srow, scol = 3, 12

    a = np.arange(am*an, dtype=np.float64).reshape(am, an)
    a = a + 1.0J * np.arange(am*an, dtype=np.float64).reshape(am, an)
    a = np.asfortranarray(a)

    gA = np.random.standard_normal((Am, An)).astype(np.float64)
    gA = gA + 1.0J * np.random.standard_normal((Am, An)).astype(np.float64)
    gA = np.asfortranarray(gA)

    dA = core.DistributedMatrix.from_global_array(gA, rank=0)

    dA = dA.np2self(a, srow, scol, rank=1)
    a1 = dA.self2np(srow, am, scol, an, rank=2)

    if rank == 2:
        assert allclose(a, a1)


if __name__ == '__main__':
    test_np2self_D()
    test_np2self_Z()
