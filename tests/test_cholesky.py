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


def test_cholesky_D():
    ## Test the Cholesky decomposition of a double precision matrix (use the
    ## default, upper half)
    ns = 317

    gA = np.random.standard_normal((ns, ns)).astype(np.float64)
    gA = np.dot(gA, gA.T)  # Make positive definite
    gA = np.asfortranarray(gA)

    dA = core.DistributedMatrix.from_global_array(gA, rank=0)

    dU = rt.cholesky(dA)
    gUd = dU.to_global_array(rank=0)

    if rank == 0:
        gUn = la.cholesky(gA)

        print gUn
        print gUd
        assert allclose(gUn, gUd)


def test_eigh_Z():
    ## Test the Cholesky decomposition of a double precision complex matrix (use the
    ## non-default, lower half)

    ns = 342

    gA = np.random.standard_normal((ns, ns)).astype(np.float64)
    gA = gA + 1.0J * np.random.standard_normal((ns, ns)).astype(np.float64)
    gA = np.dot(gA, gA.T.conj())  # Make positive definite
    gA = np.asfortranarray(gA)

    dA = core.DistributedMatrix.from_global_array(gA, rank=0)

    dU = rt.cholesky(dA, lower=True)
    gUd = dU.to_global_array(rank=0)

    if rank == 0:
        gUn = la.cholesky(gA, lower=True)

        assert allclose(gUn, gUd)
