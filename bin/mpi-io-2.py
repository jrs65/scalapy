from pyscalapack import blockcyclic

import numpy as np

from mpi4py import MPI

#from pyscalapack import core as pscore

comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

blocksize = [2, 2]
gshape = [9, 9]
pshape = [int(size**0.5), int(size**0.5)]



arr = np.arange(gshape[0]*gshape[1]).astype(np.float64).reshape(gshape)
arr.tofile("testarr_c.dat")
np.asfortranarray(arr).tofile("testarr_f.dat")



local_array_c = blockcyclic.mpi_readmatrix("testarr_c.dat", comm, gshape, np.float64, blocksize, pshape, order='C')
local_array_f = blockcyclic.mpi_readmatrix("testarr_f.dat", comm, gshape, np.float64, blocksize, pshape, order='F')

for i in range(size):
    comm.Barrier() 

    if rank == i:
        print [int(rank / pshape[1]), int(rank % pshape[1])]
        #print local_array.flags
        #print local_array

        print "Fortran ordered."
        print local_array_f
        print
        print "C ordered."
        print local_array_c
        print

blockcyclic.mpi_writematrix("testarr2_c.dat", local_array_c, comm, gshape, np.float64, blocksize, pshape, order='C')
blockcyclic.mpi_writematrix("testarr2_f.dat", local_array_f, comm, gshape, np.float64, blocksize, pshape, order='F')


if rank == 0:

    arr1_c = np.fromfile("testarr_c.dat")
    arr2_c = np.fromfile("testarr2_c.dat")

    arr1_f = np.fromfile("testarr_f.dat")
    arr2_f = np.fromfile("testarr2_f.dat")

    print "F ordered"
    print arr1_f
    print arr2_f
    print
    print "C ordered"
    print arr1_c
    print arr2_c


MPI.Finalize()



