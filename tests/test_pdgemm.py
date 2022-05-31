import numpy as np
import pytest

from mpi4py import MPI

from scalapy import core
import scalapy.lowlevel.pblas as pblas_ll
import scalapy.lowlevel as ll


comm = MPI.COMM_WORLD

rank = comm.rank
size = comm.size

allclose = lambda a, b: np.allclose(a, b, rtol=1e-4, atol=1e-6)

if size != 4:
    raise Exception("Test needs 4 processes.")


test_context = {"gridshape": (2, 2), "block_shape": (8, 8)}


def pdgemm_iter_NN(n, m, k):
    with core.shape_context(**test_context):

        gA = np.asfortranarray(np.random.standard_normal((n, k)))
        gB = np.asfortranarray(np.random.standard_normal((k, m)))

        dA = core.DistributedMatrix.from_global_array(gA, rank=0)
        dB = core.DistributedMatrix.from_global_array(gB, rank=0)
        dC = core.DistributedMatrix([n, m], dtype=np.float64)

        ll.pdgemm('N', 'N', n, m, k, 1.0, dA, dB, 0.0, dC)

        gCd = dC.to_global_array(rank=0)
        gC = np.asfortranarray(np.dot(gA, gB))

        if rank == 0:
            assert allclose(gCd, gC)
        else:
            pass


@pytest.mark.parametrize("n,m,k", [[93, 91, 92], [1001, 10, 16], [42, 43, 45], [501, 502, 601]])
def test_pdgemm(n, m, k):
    with core.shape_context(**test_context):

        gA = np.random.standard_normal((n, k)) + 1.0J * np.random.standard_normal((n, k))
        gB = np.random.standard_normal((k, m)) + 1.0J * np.random.standard_normal((k, m))
        gC = np.random.standard_normal((n, m)) + 1.0J * np.random.standard_normal((n, m))

        dA = core.DistributedMatrix.from_global_array(gA, rank=0)
        dB = core.DistributedMatrix.from_global_array(gB, rank=0)
        dC = core.DistributedMatrix.from_global_array(gC, rank=0)

        ll.pzgemm('N', 'N', n, m, k, 1.1, dA, dB, 8.0, dC)

        gCd = dC.to_global_array(rank=0)
        gC = 1.1 * np.dot(gA, gB) + 8.0 * gC

        if rank == 0:
            assert allclose(gCd, gC)
        else:
            pass
