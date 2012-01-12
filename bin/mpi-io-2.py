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



arr = np.arange(gshape[0]*gshape[1]).astype(np.float64).reshape(gshape, order='F')
arr.tofile("testarr.dat")



local_array = blockcyclic.mpi_readmatrix("testarr.dat", comm, gshape, np.float64, blocksize, pshape, order='F')

for i in range(size):
    comm.Barrier() 

    if rank == i:
        print [int(rank / pshape[1]), int(rank % pshape[1])]
        print local_array.flags
        print local_array


MPI.Finalize()



