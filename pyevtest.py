
import scarray
import numpy as np

import scipy.linalg as la

from mpi4py import MPI
comm = MPI.COMM_WORLD

scarray._blocksize = [4, 4]
gsize = [10, 10]

scarray.initmpi()

A = np.random.standard_normal(gsize).astype(np.float64)

Am = scarray.LocalMatrix.fromarray(A)

evals1, evecs1 = scarray.pdsyevd(Am)

if comm.Get_rank() == 0:
    evals2, evecs2 = la.eigh(A)

    print "=== Scalapack ==="
    print evals1

    print "=== Scipy ==="
    print evals2

    print
    print "Max diff:", np.abs((evals1 - evals2) / evals1).max()
