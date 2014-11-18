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

core.initmpi([2, 2], block_shape=[16, 16])

allclose = lambda a, b: np.allclose(a, b, rtol=1e-4, atol=1e-6)


def test_lu_D():
    ## Test the LU factorization of a real double precision matrix
    ns = 357

    gA = np.random.standard_normal((ns, ns)).astype(np.float64)
    gA = np.asfortranarray(gA)

    dA = core.DistributedMatrix.from_global_array(gA, rank=0)

    LU, ipiv = rt.lu(dA)
    gLU = LU.to_global_array(rank=0)

    # print 'Process %d has ipiv = %s' % (rank, ipiv)

    if rank == 0:
        P, L, U = la.lu(gA)
        # compare with scipy result
        assert allclose(gLU, L + U - np.eye(ns, dtype=np.float64))


def test_lu_Z():
    ## Test the LU factorization of a complex double precision matrix
    ns = 478

    gA = np.random.standard_normal((ns, ns)).astype(np.float64)
    gA = gA + 1.0J * np.random.standard_normal((ns, ns)).astype(np.float64)
    gA = np.asfortranarray(gA)

    dA = core.DistributedMatrix.from_global_array(gA, rank=0)

    LU, ipiv = rt.lu(dA)
    gLU = LU.to_global_array(rank=0)

    # print 'Process %d has ipiv = %s' % (rank, ipiv)

    if rank == 0:
        P, L, U = la.lu(gA)
        # compare with scipy result
        assert allclose(gLU, L + U - np.eye(ns, dtype=np.complex128))


if __name__ == '__main__':
    test_lu_D()
    test_lu_Z()
