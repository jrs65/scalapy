
import scarray
import scroutine

import numpy as np

import scipy.linalg as la

from mpi4py import MPI

import os
import time
import sys

os.environ['OMP_NUM_THREADS'] = '1'

comm = MPI.COMM_WORLD

def f(x):
    return np.where(x < 10, np.ones_like(x), np.zeros_like(x))

nproc = comm.Get_size()
rank = comm.Get_rank()

n = 5000
B = 256
npx = int(comm.Get_size()**0.5)
npy = npx

if len(sys.argv) > 2:
    n = int(sys.argv[1])
    B = int(sys.argv[2])

if len(sys.argv) > 3:
    npx = int(sys.argv[3])
    npy = npx

if len(sys.argv) > 4:
    npy = int(sys.argv[4])

if len(sys.argv) > 5:
    os.environ['OMP_NUM_THREADS'] = int(sys.argv[5])


scarray.initmpi(gridsize = [npx, npy], blocksize = [B, B])

A = scarray.LocalMatrix([n, n])



if rank == 0:
    print "==============================================="
    print "Decomposing  %i x %i global matrix" % (n, n)
    print
    print "Local shape: %i x %i" % A.local_shape()
    print "Blocksize:   %i x %i" % (B, B)
    print "Pgrid size:  %i x %i" % (npx, npy)
    print
    print "Number of threads: %i" % (int(os.environ['OMP_NUM_THREADS']) if 'OMP_NUM_THREADS' in os.environ else 0)
    print "==============================================="
    print 
    


if comm.Get_rank() == 0:
    print "Setting up..."
    st = time.time()


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

