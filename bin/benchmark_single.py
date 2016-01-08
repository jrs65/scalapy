from __future__ import print_function, division, absolute_import

import numpy as np

import scipy.linalg as la

import os
import time
import sys

os.environ['OMP_NUM_THREADS'] = '1'

def f(x):
    #return np.where(x < 10, np.ones_like(x), np.zeros_like(x))
    return np.exp(-(x**2) / (2.0*(10.0**2))) + 1e-10* np.where(x == 0, np.ones_like(x), np.zeros_like(x))


n = 5000

n = int(sys.argv[1])
nthread = int(sys.argv[2])

os.environ['OMP_NUM_THREADS'] = repr(nthread)


print("===============================================")
print("Decomposing  %i x %i global matrix" % (n, n))
print()
print("Number of threads: %i" % (int(os.environ['OMP_NUM_THREADS']) if 'OMP_NUM_THREADS' in os.environ else 0))
print("===============================================")
print()

st = time.time()
# Construct array of distances between indices (taking into account
# periodicity).
ri, ci = np.indices((n,n))
da = np.abs(ri-ci)
del ri, ci

na = n - da
da = np.where(da < na, da, na)

del na

A = f(da)

del da

#A.local_matrix[:,:] = np.random.standard_normal(A.local_shape())

#================

et = time.time()
print("Done. Time: ", et-st)
st = time.time()
print("Starting eigenvalue solve...")

evals1, evecs1 = la.eigh(A)

et = time.time()
print("Done. Time: ", et-st)
del evecs1

evtime = et - st

#================

st = time.time()
print("Starting Cholesky...")

U = la.cholesky(A)

et = time.time()
print("Done. Time: ", et-st)
chtime = et - st

#================

st = time.time()
print("Starting matrix multiply...")

A2 = np.dot(U.T, U)

et = time.time()
print("Done. Time: ", et-st)
mltime = et - st

#================

st = time.time()
print("Starting verification...")


# Calculate eigenvalues by fourier transform.
x = np.arange(n, dtype=np.float64)
px = np.where(x < n-x, x, n-x)
evals2 = np.sort(np.fft.fft(f(px)).real)

me = (A==A2).all()

print("Max diff:", np.abs((evals1 - evals2) / evals1).max())

print("A == A2:", me)
#print "Max diff A, rnk 0:", np.abs(A - A2.local_matrix).max() / np.abs(A.local_matrix).max()

#bfile = bfile + "_%i_%i_%i_%i_%i.dat" % (n, B, npx, npy, nthread)

print("%i %i %g %g %g\n" % (n, nthread, evtime, chtime, mltime))
