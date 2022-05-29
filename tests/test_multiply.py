import numpy as np

from mpi4py import MPI

from scalapy import core


comm = MPI.COMM_WORLD

rank = comm.rank
size = comm.size

if size != 4:
    raise Exception("Test needs 4 processes.")

test_context = {"gridshape": (2, 2), "block_shape": (3, 3)}

allclose = lambda a, b: np.allclose(a, b, rtol=1e-4, atol=1e-6)


def test_mul():
    """Test multiply method of a distributed matrix"""
    with core.shape_context(**test_context):

        ms, ns = 5, 14

        gA = np.random.standard_normal((ms, ns)).astype(np.float64)
        gA = np.asfortranarray(gA)
        dA = core.DistributedMatrix.from_global_array(gA, rank=0)

        gB = np.random.standard_normal((ms, ns)).astype(np.float64)
        gB = np.asfortranarray(gB)
        dB = core.DistributedMatrix.from_global_array(gB, rank=0)

        dC = dA * dB
        gC = dC.to_global_array(rank=0)

        a = np.random.standard_normal(ns).astype(np.float64)
        comm.Bcast(a, root=0) # ensure all process have the same data
        dD = dA * a
        gD = dD.to_global_array(rank=0)

        alpha = 2.345
        dE = dA * alpha
        gE = dE.to_global_array(rank=0)

        if rank == 0:
            assert allclose(gA * gB, gC)
            assert allclose(gA * a, gD)
            assert allclose(gA * alpha, gE)
