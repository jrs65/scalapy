
import scarray
import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD

blocksize = [4, 4]

scarray._blocksize = blocksize

scarray.initmpi()

x, y = np.meshgrid(np.arange(10, dtype=np.float64), np.arange(10, dtype=np.float64))

xm = scarray.LocalMatrix.fromarray(x)
ym = scarray.LocalMatrix.fromarray(y)

xm2 = scarray.LocalMatrix(globalsize = [10, 10])

scarray.index_array(10, 4, 0, 2)

yi, xi = xm2.indices()
xm2.local_matrix[:,:] = xi

ym2 = scarray.LocalMatrix(globalsize = [10, 10])
ym2.local_matrix[:,:] = yi

print "xm == xm2:", scarray.matrix_equal(xm, xm2)
print "ym == ym2:", scarray.matrix_equal(ym, ym2)
