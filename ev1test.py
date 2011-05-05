
import numpy as np

import scarray


m1 = np.identity(2000, dtype=np.float64)

m2 = scarray.matrix_pagealign(m1, [1000, 1000])

m2.reshape(-1, order='A').tofile("mat1pa.dat")



m2a = np.fromfile("mat1pa.dat", dtype=np.float64)

m3 = scarray.matrix_from_pagealign(m2a, [2000, 2000], [1000, 1000])



if (m1 == m3).all():
    print "Matrices correct."
else:
    print "Eeek."
