
import scalapy
import numpy as np

from mpi4py import MPI

import os
import time
import sys

#os.environ['OMP_NUM_THREADS'] = '1'

comm = MPI.COMM_WORLD


def f(x):
    #return np.where(x < 10, np.ones_like(x), np.zeros_like(x))
    return np.exp(-(x**2) / (2.0*(10.0**2))) + 1e-10 * np.where(x == 0, np.ones_like(x), np.zeros_like(x))

nproc = comm.Get_size()
rank = comm.Get_rank()

n = 5000
B = 256
npx = int(comm.Get_size()**0.5)
npy = npx

if len(sys.argv) < 6:
    raise Exception("Too few arguments.")

n = int(sys.argv[1])
B = int(sys.argv[2])
npx = int(sys.argv[3])
npy = int(sys.argv[4])
nthread = int(sys.argv[5])

bfile = sys.argv[6]


os.environ['OMP_NUM_THREADS'] = repr(nthread)

scalapy.initmpi([npx, npy], block_shape=[B, B])

A = scalapy.DistributedMatrix([n, n])


if rank == 0:
    print "==============================================="
    print "Decomposing  %i x %i global matrix" % (n, n)
    print
    print "Local shape: %i x %i" % A.local_shape
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

A.local_array[:, :] = f(da)
#A.local_array[:,:] = np.random.standard_normal(A.local_shape())

comm.Barrier()

if comm.Get_rank() == 0:
    et = time.time()
    print "Done. Time: ", et-st
    st = time.time()
    print "Starting eigenvalue solve..."

evals1, evecs1 = scalapy.eigh(A, overwrite_a=False)

comm.Barrier()

if comm.Get_rank() == 0:
    et = time.time()
    print "Done. Time: ", et-st

    evtime = et - st

comm.Barrier()

if comm.Get_rank() == 0:
    st = time.time()
    print "Starting Cholesky..."

U = scalapy.cholesky(A, overwrite_a=False)

comm.Barrier()

if comm.Get_rank() == 0:
    et = time.time()
    print "Done. Time: ", et-st

    chtime = et - st

    st = time.time()
    print "Starting matrix multiply..."


A2 = scalapy.dot(U, U, transA='T')


comm.Barrier()

if comm.Get_rank() == 0:
    et = time.time()
    print "Done. Time: ", et-st

    mltime = et - st

    st = time.time()
    print "Starting verification..."


# Calculate eigenvalues by fourier transform.
x = np.arange(n, dtype=np.float64)
px = np.where(x < n-x, x, n-x)
evals2 = np.sort(np.fft.fft(f(px)).real)


if comm.Get_rank() == 0:
    print "Max diff:", np.abs((evals1 - evals2) / evals1).max()

    print "Max diff A, rnk 0:", np.abs(A.local_array - A2.local_array).max() / np.abs(A.local_array).max()

    #bfile = bfile + "_%i_%i_%i_%i_%i.dat" % (n, B, npx, npy, nthread)

    with open(bfile, "w+") as f:
        line = "%i %i %i %i %i %g %g %g\n" % (n, B, npx, npy, nthread, evtime, chtime, mltime)
        f.write(line)


