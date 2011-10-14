
import pyscalapack as pysc
import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD

#pysc._blocksize = blocksize

pysc.initmpi(blocksize = [4, 4])

x, y = np.meshgrid(np.arange(10, dtype=np.complex128), np.arange(10, dtype=np.complex128))

xm = pysc.DistributedMatrix.fromarray(x)
ym = pysc.DistributedMatrix.fromarray(y)

xm2 = pysc.DistributedMatrix(globalsize = [10, 10], dtype=np.complex128)

pysc.index_array(10, 4, 0, 2)

yi, xi = xm2.indices()
xm2.local_array[:,:] = xi

ym2 = pysc.DistributedMatrix(globalsize = [10, 10], dtype=np.complex128)
ym2.local_array[:,:] = yi

print "xm == xm2:", pysc.matrix_equal(xm, xm2)
print "ym == ym2:", pysc.matrix_equal(ym, ym2)

#comm.Barrier()
#if comm.Get_rank() == 0:
#    print x
#    print y


#for i in range(comm.Get_size()):
#    comm.Barrier()
#    if i == comm.Get_rank():
#        print "=========="
#        print xm.dtype.itemsize
#        print xm.local_array
#        print xm2.dtype.itemsize
#        print xm2.local_array
#
#        print ym.local_array
#        print ym2.local_array
#
#        print "=========="

