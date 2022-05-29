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


def test_inv_D():
    """Test inverse computation of a real double precision distributed matrix"""
    with core.shape_context(**test_context):

        ns = 353

        gA = np.random.standard_normal((ns, ns)).astype(np.float64)
        gA = np.asfortranarray(gA)

        dA = core.DistributedMatrix.from_global_array(gA, rank=0)

        invA, ipiv = rt.inv(dA)
        ginvA = invA.to_global_array(rank=0)

        # print 'Process %d has ipiv = %s' % (rank, ipiv)

        if rank == 0:
            assert allclose(np.dot(ginvA, gA), np.eye(ns, dtype=np.float64))
            assert allclose(ginvA, la.inv(gA)) # compare with scipy result


def test_inv_Z():
    """Test inverse computation of a complex double precision distributed matrix"""
    with core.shape_context(**test_context):

        ns = 521

        gA = np.random.standard_normal((ns, ns)).astype(np.float64)
        gA = gA + 1.0J * np.random.standard_normal((ns, ns)).astype(np.float64)
        gA = np.asfortranarray(gA)

        dA = core.DistributedMatrix.from_global_array(gA, rank=0)

        invA, ipiv = rt.inv(dA)
        ginvA = invA.to_global_array(rank=0)

        # print 'Process %d has ipiv = %s' % (rank, ipiv)

        if rank == 0:
            assert allclose(np.dot(ginvA, gA), np.eye(ns, dtype=np.complex128))
            assert allclose(ginvA, la.inv(gA)) # compare with scipy result
