
import scarray
import scroutine

import numpy as np

import scipy.linalg as la

from mpi4py import MPI

import os
import time

#os.environ['OMP_NUM_THREADS'] = '1'

comm = MPI.COMM_WORLD

def f(x):
    return np.where(x < 10, np.ones_like(x), np.zeros_like(x))

if comm.Get_rank() == 0:
    print "Setting up..."
    st = time.time()

n = 5000
B = 256
gsize = [n, n]

nproc = comm.Get_size()
if nproc == 12:
    grid = [4,3]
elif nproc == 16:
    grid = [4,4]
else:
    grid = None

scarray.initmpi(gridsize = grid, blocksize = [B, B])

np.random.seed(0)

A = scarray.LocalMatrix(gsize)

# Construct array of distances between indices (taking into account
# periodicity).
ri, ci = A.indices()
da = np.abs(ri-ci)
na = n - da
da = np.where(da < na, da, na)

A.local_matrix[:,:] = f(da)
#A.local_matrix[:,:] = np.random.standard_normal(A.local_shape())

comm.Barrier()

if comm.Get_rank() == 0:
    et = time.time()
    print "Done. Time: ", et-st
    st = time.time()
    print "Starting eigenvalue solve..."

evals1, evecs1 = scroutine.pdsyevd(A)

comm.Barrier()

if comm.Get_rank() == 0:
    et = time.time()
    print "Done. Time: ", et-st

# Calculate eigenvalues by fourier transform.
x = np.arange(n, dtype=np.float64)
px = np.where(x < n-x, x, n-x)
evals2 = np.sort(np.fft.fft(f(px)).real)

if comm.Get_rank() == 0:
    print "Max diff:", np.abs((evals1 - evals2) / evals1).max()

