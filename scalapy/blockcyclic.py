"""
=======================================================
Blockcyclic Utilities (:mod:`~scalapy.blockcyclic`)
=======================================================

A set of utilities for calculating the packing in block cyclic matrix
distributions.

Routines
========

.. autosummary::
    :toctree: generated/

    numrc
    indices_rc

    num_blocks
    num_c_blocks
    num_lblocks
    num_c_lblocks 
    partial_last_block
"""
from __future__ import print_function

import numpy as np

from mpi4py import MPI


def ceildiv(x, y):
    """Round to ceiling division."""
    return ((int(x) - 1) / int(y) + 1)


def pid_remap(p, p0, P):
    return ((p + P - p0) % P)


def num_c_blocks(N, B):
    """Number of complete blocks globally.

    Parameters
    ----------
    N : integer
        Number of elements on the side.
    B : integer
        Block length.

    Returns
    -------
    num : integer
    """
    return int(N / B)


def num_blocks(N, B):
    """Total number of blocks globally (complete or not).

    Parameters
    ----------
    N : integer
        Number of elements on the side.
    B : integer
        Block length.

    Returns
    -------
    num : integer
    """
    return ceildiv(N, B)


def num_c_lblocks(N, B, p, P):
    """Number of complete blocks locally.

    Parameters
    ----------
    N : integer
        Number of elements on the side.
    B : integer
        Block length.
    p : integer
        Process index.
    P : integer
        Number of processes on the side.

    Returns
    -------
    num : integer
    """
    nbc = num_c_blocks(N, B)
    return int(nbc / P) + int(1 if ((nbc % P) > p) else 0)


def num_lblocks(N, B, p, P):
    """Total number of local blocks.

    Parameters
    ----------
    N : integer
        Number of elements on the side.
    B : integer
        Block length.
    p : integer
        Process index.
    P : integer
        Number of processes on the side.

    Returns
    -------
    num : integer
    """
    nb = num_blocks(N, B)
    return int(nb / P) + int(1 if ((nb % P) > p) else 0)


def partial_last_block(N, B, p, P):
    """Is the last local block partial?

    Parameters
    ----------
    N : integer
        Number of elements on the side.
    B : integer
        Block length.
    p : integer
        Process index.
    P : integer
        Number of processes on the side.

    Returns
    -------
    partial : boolean
    """
    return ((N % B > 0) and ((num_c_blocks(N, B) % P) == p))


def numrc(N, B, p, P):
    """The number of rows/columns of the global array local to the process.

    Parameters
    ----------
    N : integer
        Number of elements on the side.
    B : integer
        Block length.
    p : integer
        Process index.
    P : integer
        Number of processes on the side.

    Returns
    -------
    num : integer

    Examples
    --------

    >>> numrc(5, 2, 0, 2)
    3

    >>> numrc(5, 2, 1, 2)
    2
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

    Parameters
    ----------
    N : integer
        Number of elements on the side.
    B : integer
        Block length.
    p : integer
        Process index.
    P : integer
        Number of processes on the side.

    Returns
    -------
    indices : np.ndarray[int32]    
        Indices of the side that are local to this process.

    Examples
    --------
    Short example:

    >>> indices_rc(5, 2, 0, 2)
    np.array([0, 1, 4])

    >>> indices_rc(5, 2, 1, 2)
    np.array([2, 3])
    """

    nt = numrc(N, B, p, P)
    nb = num_c_lblocks(N, B, p, P)

    ind = np.zeros(nt, dtype='int')

    ind[:(nb*B)] = ((np.arange(nb)[:, np.newaxis] * P + p)*B +
                    np.arange(B)[np.newaxis, :]).flatten()

    if (nb * B < nt):
        ind[(nb*B):] = (nb*P+p)*B + np.arange(nt - nb*B)

    return ind


def localize_indices(global_indices, B, P):
    """Given an array of "global indices", compute the (rank, local index) pair corresponding to each global index.
    
    Parameters
    ----------
    global_indices : integer-valued array
        Array of global indices
    B : integer
        Block length.
    P : integer
        Number of processes on the side.

    Returns
    -------
    rank : integer-valued array
        Array of ranks (between 0 and P)
    local_indices : integer-valued array
        Array of local indices
    """

    global_indices = np.array(global_indices)
    assert np.issubdtype(global_indices.dtype, np.integer)
    assert np.all(global_indices >= 0)
    assert B > 0
    assert P > 0

    t = np.divide(global_indices, B)
    u = np.divide(t, P)
    return (t-u*P, global_indices+B*(u-t))


def mpi_readmatrix(fname, comm, gshape, dtype, blocksize, process_grid, order='F', displacement=0):
    """Distribute a block cyclic matrix read from a file (using MPI-IO).

    The order flag specifies in which order (either C or Fortran) the array is
    on disk. Importantly the returned `local_array` is ordered the *same* way.
    
    Parameters
    ----------
    fname : string
        Name of file to read.
    comm : mpi4py.MPI.COMM
        MPI communicator to use. Must match with the one used by BLACS (if using
        Scalapack).
    gshape : (nrows, ncols)
        Shape of the global matrix.
    blocksize : (blockrows, blockcols)
        Blocking size for distribution.
    process_grid : (prows, pcols)
        The shape of the process grid. Must be the same total size as
        comm.Get_rank(), and match the BLACS grid (if using Scalapack).
    order : 'F' or 'C', optional
        Is the matrix on disk is 'F' (Fortran/column major), or 'C' (C/row
        major) order. Defaults to Fortran ordered.
    displacement : integer, optional
        Use a displacement from the start of the file. That is ignore the first
        `displacement` bytes.

    Returns
    -------
    local_array : np.ndarray
        The section of the array local to this process.
    """
    if dtype not in _typemap:
        raise Exception("Unsupported type.")

    # Get MPI type
    mpitype = _typemap[dtype]


    # Sort out F, C ordering
    if order not in ['F', 'C']:
        raise Exception("Order must be 'F' (Fortran) or 'C'")

    # Set file ordering
    mpiorder = MPI.ORDER_FORTRAN if order=='F' else MPI.ORDER_C 

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
        raise Exception("Strange mismatch is local shape size.")


    # Create the local array
    local_array = np.empty(lshape, dtype=dtype, order=order)

    # Open the file, and read out the segments
    f = MPI.File.Open(comm, fname, MPI.MODE_RDONLY)
    f.Set_view(displacement, mpitype, darr, "native")
    f.Read_all(local_array)
    f.Close()

    return local_array



    
def mpi_writematrix(fname, local_array, comm, gshape, dtype,
                    blocksize, process_grid, order='F', displacement=0):
    
    """Write a block cyclic distributed matrix to a file (using MPI-IO).

    The order flag specifies in which order (either C or Fortran) the array
    should be on on disk. Importantly the input `local_array` *must* be ordered
    in the same way.
    
    Parameters
    ----------
    fname : string
        Name of file to read.
    local_array : np.ndarray
        The array to write.
    comm : mpi4py.MPI.COMM
        MPI communicator to use. Must match with the one used by BLACS (if using
        Scalapack).
    gshape : (nrows, ncols)
        Shape of the global matrix.
    blocksize : (blockrows, blockcols)
        Blocking size for distribution.
    process_grid : (prows, pcols)
        The shape of the process grid. Must be the same total size as
        comm.Get_rank(), and match the BLACS grid (if using Scalapack).
    order : 'F' or 'C', optional
        Is the matrix on disk is 'F' (Fortran/column major), or 'C' (C/row
        major) order. Defaults to Fortran ordered.
    displacement : integer, optional
        Use a displacement from the start of the file. That is ignore the first
        `displacement` bytes.

    """
    
    if dtype not in _typemap:
        raise Exception("Unsupported type.")

    # Get MPI type
    mpitype = _typemap[dtype]


    # Sort out F, C ordering
    if order not in ['F', 'C']:
        raise Exception("Order must be 'F' (Fortran) or 'C'")

    mpiorder = MPI.ORDER_FORTRAN if order=='F' else MPI.ORDER_C 


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

    # Check to see if they type has the same shape.
    if local_array.size != darr.Get_size() / mpitype.Get_size():
        raise Exception("Local array size is not consistent with array description.")

    # Length of filename required for write (in bytes).
    filelength = displacement + gshape[0]*gshape[1]

    print(filelength, darr.Get_size())

    # Open the file, and read out the segments
    f = MPI.File.Open(comm, fname, MPI.MODE_RDWR | MPI.MODE_CREATE)

    # Preallocate to ensure file is long enough for writing.
    f.Preallocate(filelength)

    # Set view and write out.
    f.Set_view(displacement, mpitype, darr, "native")
    f.Write_all(local_array)
    f.Close()
