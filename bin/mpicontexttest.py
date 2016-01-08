from __future__ import print_function, division, absolute_import

from mpi4py import MPI
import numpy as np

import scipy.linalg as la

from pyscalapack import core as pscore
from pyscalapack import routines as psr

# Set parameters of the test matrix
n = 256
b = 16

# Set up some world properties
world_group = MPI.COMM_WORLD.Get_group()
world_comm = MPI.COMM_WORLD
rank = world_comm.rank

# Split the world_group into subgroups of four processes
group_index = rank // 4
group_rank = rank % 4
group_indices = np.arange(4) + group_index*4
new_group = world_group.Incl(group_indices)
new_comm = world_comm.Create(new_group)

# Set the default blocksize
pscore._blocksize = [b, b]

# Process 0 creates the test matrices for each group which are identical up to
# rescaling. The rescaling is such that the greatest eigenvalue is equal to the
# group index.
if rank == 0:
    mat = np.random.standard_normal((n, n))
    mat = mat + mat.T
    mat = mat / la.eigvalsh(mat).max()

    for i in range(world_comm.size):
        np.save("testarr_%i.npy" % i, mat*(i+1))

world_comm.Barrier()

# Create the Process Context
context = pscore.ProcessContext(comm=new_comm, gridsize=[2,2])

# Create a distributed matrix and find its eigenvalues
dm = pscore.DistributedMatrix.from_npy("testarr_%i.npy" % group_index, context=context)
evals, evecs = psr.pdsyevd(dm)

# Print out results for testing.
for i in range(world_comm.size):
    world_comm.Barrier()

    if rank == i:
        print("Group %i. Group rank %i. Rank %i. " % (group_index, group_rank, rank))
        print(evals[-10:])
        print()
