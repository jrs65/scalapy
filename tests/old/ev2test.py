
import numpy as np

import scipy.linalg as la

import scarray

import os

matfile = "mat.dat"
evecsfile = "evecs.dat"
evalsfile = "evals.dat"

Br = 1000
Bc = 1000

Pr = 2
Pc = 2


#m1 = np.identity(2000, dtype=np.float64)
print "Generating matrix"
m1 = np.random.standard_normal((2000,2000))
m1 = 0.5*(m1 + m1.T)

m2 = scarray.matrix_pagealign(m1, [Br, Bc])

m2.reshape(-1, order='A').tofile(matfile)

print "Running ScaLapack"
os.system("mpirun -np %i ./evtest %i %i %i %i %i %i %s %s %s" % (Pr*Pc, m1.shape[0], m1.shape[1], Br, Bc, Pr, Pc, matfile, evalsfile, evecsfile))

print "Done Scalapack."
print "Reading..."
evecst = np.fromfile(evecsfile, dtype=np.float64)

evecs1 = scarray.matrix_from_pagealign(evecst, m1.shape, [Br, Bc])

evals1 = np.fromfile(evalsfile, dtype=np.float64)

print "Done."
print "Numpy EV"

evals2, evecs2 = la.eigh(m1)

