
import scarray
import scroutine

import numpy as np

import scipy.linalg as la

from mpi4py import MPI

import time

comm = MPI.COMM_WORLD

def f(x):
    return np.where(np.abs(x) < 10, np.ones_like(x), np.zeros_like(x))

if comm.Get_rank() == 0:
    print "Setting up..."
    st = time.time()

n = 1000
scarray._blocksize = [100, 100]
gsize = [n, n]

scarray.initmpi()

np.random.seed(0)

A = scarray.LocalMatrix(gsize)

ri, ci = A.indices()

A.local_matrix[:,:] = f(ri-ci)

if comm.Get_rank() == 0:
    et = time.time()
    print "Done. Time: ", et-st
    st = time.time()
    print "Starting eigenvalue solve..."

evals1, evecs1 = scroutine.pdsyevd(A)

if comm.Get_rank() == 0:
    et = time.time()
    print "Done. Time: ", et-st


x = np.arange(n, dtype=np.float64)
px = np.where(x < n-x, x, n-x)
evals2 = np.sort(np.fft.fft(f(px)).real)

if comm.Get_rank() == 0:
    print "Max diff:", np.abs((evals1 - evals2) / evals1).max()
