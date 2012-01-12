import numpy as np

from mpi4py import MPI

def ceildiv(x, y):
    """Round to ceiling division."""
    return ((int(x) - 1) / int(y) + 1)

def pid_remap(p, p0, P):
    return ((p + P - p0) % P)

def num_c_blocks(N, B):
    """Number of complete blocks globally."""
    return int(N / B)

def num_blocks(N, B):
    """Total number of blocks globally."""
    return ceildiv(N, B)

def num_c_lblocks(N, B, p, P):
    """Number of complete blocks locally."""
    nbc = num_c_blocks(N, B)
    return int(nbc / P) + int(1 if ((nbc % P) > p) else 0)


def num_lblocks(N, B, p, P):
    """Total number of local blocks."""
    nb = num_blocks(N, B)
    return int(nb / P) + int(1 if ((nb % P) > p) else 0)

def partial_last_block(N, B, p, P):
    """Is the last local block partial?"""
    return ((N % B > 0) and ((num_c_blocks(N, B) % P) == p))


def num_rstride(N, B, stride):
    """Length of block strided row."""
    return num_blocks(N, B) * stride


#size_t stride_page(size_t B, size_t itemsize) {
#
#  size_t pl;
#
#  pl = (size_t) sysconf (_SC_PAGESIZE) / itemsize;
#  return ceildiv(B, pl) * pl;
#}


#size_t num_rpage(size_t N, size_t B, size_t itemsize) {
#  return num_rstride(N, B, stride_page(B, itemsize));
#}


def numrc(N, B, p, P):
    """The number of rows/columns of the global array local to the process.
    """
    
    # Number of complete blocks owned by the process.
    nbp = num_c_lblocks(N, B, p, P)
    
    # Number of entries of complete blocks owned by process.
    n = nbp * B
    
    # If this process owns an incomplete block, then add the number of entries.
    if partial_last_block(N, B, p, P):
        n += N % B

    return n


def indices_rc(N, B, p, P):
    """The indices of the global array local to the process.
    """
    
    nt = numrc(N, B, p, P)
    nb = num_c_lblocks(N, B, p, P)

    ind = np.zeros(nt)

    ind[:(nb*B)] = ((np.arange(nb)[:, np.newaxis] * P + p)*B +
                    np.arange(B)[np.newaxis, :]).flatten()

    if (nb * B < nt):
        ind[(nb*B):] = (nb*P+p)*B + np.arange(nt - nb*P)

    return ind






def mpi_readmatrix(fname, comm, gshape, dtype, blocksize, process_grid, order='F'):

    # Map numpy type into MPI type
    typemap = { np.float32 : MPI.FLOAT,
                np.float64 : MPI.DOUBLE,
                np.complex128 : MPI.COMPLEX16 }

    if dtype not in typemap:
        raise Exception("Unsupported type.")

    mpitype = typemap[dtype]


    # Sort out F, C ordering
    if order not in ['F', 'C']:
        raise Exception("Order must be 'F' (Fortran) or 'C'")

    mpiorder = MPI.ORDER_FORTRAN # Check what we need to do here for loading C
                                 # ordered files.


    # Get MPI process info
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Check process grid shape
    if size != process_grid[0]*process_grid[1]:
        raise Exception("MPI size does not match process grid.")



    # Create distributed array view.
    darr = mpitype.Create_darray(size, rank, gshape,
                                 [MPI.DISTRIBUTE_CYCLIC, MPI.DISTRIBUTE_CYCLIC],
                                 blocksize, process_grid, mpiorder)
    darr.Commit()

    # Get shape of loal segment
    process_position = [int(rank / process_grid[1]), int(rank % process_grid[1])]
    lshape = map(numrc, gshape, blocksize, process_position, process_grid)

    # Check to see if they type has the same shape.
    if lshape[0]*lshape[1] != darr.Get_size() / mpitype.Get_size():
        print lshape[0]*lshape[1], darr.Get_size(), mpitype.Get_size(), process_position
        raise Exception("Strange mismatch is local shape size.")


    # Create the local array
    local_array = np.empty(lshape, dtype=dtype, order=order)

    # Open the file, and read out the segments
    f = MPI.File.Open(comm, fname, MPI.MODE_RDONLY)
    f.Set_view(0, mpitype, darr, "native")
    f.Read_all(local_array)
    f.Close()

    return local_array

    

def bc_matrix_forward(src, blocksize, process, process_grid, order='F', dest=None):

    gshape = src.shape

    lshape = map(numrc, gshape, blocksize, process, process_grid)

    if dest is None:
        dest = np.empty(lshape, dtype=src.dtype, order='F')

    cblocks = map(num_c_blocks, gshape, blocksize)
    lcblocks = map(num_c_blocks, gshape, blocksize, process, process_grid)
    partial = map(partial_last_block, gshape, blocksize, process, process_grid)    

    clen = cblocks[0]*blocksize[0], cblocks[1]*blocksize[1]

    q1 = src[:clen[0], :clen[1]].reshape((blocksize[0], cblocks[0], blocksize[1], cblocks[1]))
    q2 = src[:clen[0], clen[1]:].reshape((blocksize[0], cblocks[0], -1))
    q3 = src[clen[0]:, :clen[1]].reshape((-1, blocksize[1], cblocks[1]))
    q4 = src[clen[0]:, clen[1]:].reshape((gshape[0] - clen[0], gshape[1] - clen[1]))
    
    #q1[:, process[0]::process_grid[0], :, process[1]::process_grid[1]].reshape()
        
