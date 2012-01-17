import os
import os.path
import sys

import numpy as np
cimport numpy as np

from mpi4py import MPI
import npyutils

import blockcyclic

from libc.stddef cimport size_t
from libc.stdlib cimport malloc, free

## Import Unix low level open/close
from posix.fcntl cimport *
from posix.unistd cimport *


## Import routine for pre-allocating files
cdef extern from "fcntl.h":
    int posix_fallocate(int fd, off_t offset, off_t len)

## Import flags for setting permissions
cdef extern from "sys/stat.h":
    enum: S_IRUSR
    enum: S_IWUSR
    #mode_t umask(mode_t mask)


from blacs cimport *
from bcutil cimport *

from mpi4py cimport MPI


# For some reason umask related stuff is not working. grr..
#cdef mode_t read_umask():
#    cdef mode_t mask
#    mask = umask (0);
#    umask (mask);
#    return mask


_context = None
_blocksize = None


def initmpi(gridsize = None, blocksize = None):
    r"""Initialise Scalapack on the current process.

    This routine sets up the BLACS grid, and sets the default context
    for this process.

    Parameters
    ----------
    gridsize : array_like, optional
        A two element list (or other tuple etc), containing the
        requested shape for the process grid e.g. [nprow, npcol]. If
        None (default) set a square grid using the maximum number of
        processes.
    blocksize : array_like, optional
        The default blocksize for new arrays. A two element, [brow,
        bcol] list.
    """

    
    # Setup the default context
    _context = ProcessContext(gridsize=gridsize)

    # Set default blocksize
    _blocksize = blocksize



    

    
class ProcessContext(object):
    r"""Stores information about an MPI/BLACS process.

    Attributes
    ----------
    num_rows, num_cols : integer
        The size of the process grid.
    row, col : integer
        The position in the process grid.
    mpi_rank, mpi_size : integer
        The index, and number of MPI processes.
    blacs_context : integer
        The BLACS context.
    """
    num_rows = 1
    num_cols = 1

    row = 0
    col = 0

    mpi_comm = None

    blacs_context = 0

    def __init__(self, comm=None, gridsize=None):
        """Construct a BLACS context for the current process.
        
        Creates a BLACS context for the given MPI Communicator.
        
        Parameters
        ----------
        comm : mpi4py.MPI.Comm, optional
            The MPI communicator to create a BLACS context for. If comm=None,
            then use MPI.COMM_WORLD instead.
        
        gridsize : array_like, optional
            A two element list (or other tuple etc), containing the
            requested shape for the process grid e.g. [nprow, npcol]. If
            None (default) set a square grid using the maximum number of
            processes.
        """
        cdef int ictxt, row, col, nrows, ncols
        
        # MPI setup
        if comm is None:
            comm = MPI.COMM_WORLD
        
        ## Attempt to set a gridsize
        if gridsize is None:
            side = int((comm.size*1.0)**0.5)
            gridsize = [side, side]
            print "No grid size supplied. Creating grid of size: %i x %i" % gridsize

        gs = gridsize[0]*gridsize[1]
        if gs != comm.size:
            raise Exception("Requested gridsize must be equal to the number of MPI processes.")
        #if gs < comm.size:
            # Replace communicator with one containing only the processes that
            # will be in the grid.
            #comm = comm.Create(comm.Get_group().Incl(range(gs)))

        self.mpi_comm = comm
        
        ictxt = Csys2blacs_handle(<MPI_Comm>(<MPI.Comm>comm).ob_mpi)

        Cblacs_gridinit(&ictxt, "Row", gridsize[0], gridsize[1])
        Cblacs_gridinfo(ictxt, &nrows, &ncols, &row, &col)

        # Fill out ProcessContext properties
        self.blacs_context = ictxt

        self.num_rows = nrows
        self.num_cols = ncols

        self.row = row
        self.col = col




def matrix_pagealign(mat, blocksize):
    r"""Page aligns the blocks in a matrix, and makes Fortran ordered.

    Parameters
    ==========
    mat : ndarray
        The matrix to page align (can either by C or Fortran ordered).
    blocksize : array_like
        The blocksize, the first and second elements correspond to the
        row and column blocks respectively.

    Returns
    =======
    m2 : ndarray
        The page aligned matrix.
    """
    #cdef np.ndarray[np.float64_t, ndim=2] m1
    #cdef np.ndarray[np.float64_t, ndim=2] m2

    m1 = np.asfortranarray(mat)

    Nr, Nc = mat.shape
    Br, Bc = blocksize

    nr = num_rpage(Nr, Br, mat.dtype.itemsize)

    m2 = np.empty((nr, Nc), order='F')

    bc2d_copy_pagealign(np_data(m1), np_data(m2), mat.dtype.itemsize, Nr, Nc, Br, Bc)

    return m2



def matrix_from_pagealign(matp, size, blocksize):
    r"""Page aligns the blocks in a matrix, and makes Fortran ordered.

    Parameters
    ==========
    matp : ndarray
        The matrix to page align (can either by C or Fortran ordered).
    blocksize : array_like
        The blocksize, the first and second elements correspond to the
        row and column blocks respectively.

    Returns
    =======
    m2 : ndarray
        The page aligned matrix.
    """
    #cdef np.ndarray[np.float64_t, ndim=1] m1
    #cdef np.ndarray[np.float64_t, ndim=2] m2

    #m1 = matp.flatten()
    m1 = matp.reshape(-1, order='A')

    Nr, Nc = size
    Br, Bc = blocksize

    nr = num_rpage(Nr, Br, matp.dtype.itemsize)

    if np.size(m1) < nr*Nc:
        raise Exception("Source matrix not long enough. Is %i x %i, should be %i x %i." % (matp.shape[0], matp.shape[1], nr, Nc))

    m2 = np.empty((Nr, Nc), order='F')

    bc2d_from_pagealign(np_data(m1), np_data(m2), matp.dtype.itemsize, Nr, Nc, Br, Bc)

    return m2



def index_array(N, B, p, P):
    r"""Which indices of the global array are local to this process.

    Parameters
    ----------
    N : integer
        Size of global array
    B : integer
        Size of blocks
    p : integer
        Index of process along this side.
    P : integer
        Number of processes on this side.

    Returns
    -------
    index_array : ndarray
        An array of integers giving the global positions that would be
        stored locally.
    """
    n = numrc(N, B, p, 0, P)
    ia = np.zeros(n, dtype=np.int32)
    rv = indices_rc(N, B, p, P, <int *>np_data(ia))

    return ia
    




cdef char * np_data(np.ndarray a):
    return <char *>a.data



def ensure_filelength(fname, length):
    r"""Make sure a file is larger than a given size.

    This will ensure that a file exists, and has a size of at least
    `length`. If this can't be done, an exception will be raised.

    Parameters
    ----------
    fname : string
        The name of the file.
    length : integer
        The requested size of the file in bytes.
    """
    cdef int fd, res

    # I've no idea why umask is not being set correctly.
    fd = open(fname, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR)
    if fd == -1:
        raise IOError("Error opening file: %s" % fname)

    res = posix_fallocate(fd, 0, <size_t>length)
    close(fd)
    
    if res != 0:
        raise IOError("Allocating file %s length %i bytes failed." % (fname, length))

    st = os.stat(fname)
    if st.st_size < length:
        raise IOError("Allocating file %s length %i bytes failed in some odd way.")
    

    







cdef class DistributedMatrix(object):
    r"""A matrix distributed over multiple MPI processes.

    Attributes
    ----------
    Nr, Nc : integer, readonly
        The number of rows and cokumns of the global matrix.
    Br, Bc : integer, readonly
        The block shape.
    
    """

    property local_array:
        r"""The local, block-cyclic packed segment of the matrix.

        This is an ndarray and is readonly. However, only the
        reference is readonly, the array itself can be modified in
        place.
        """
        def __get__(self):
            return self._local_array

    
    property desc:
        r"""The Scalapack array descriptor. See [1]_. Returned as an integer
        ndarray and is readonly.

        .. [1] http://www.netlib.org/scalapack/slug/node77.html
        """
        def __get__(self):
            return self._desc.copy()


    property context:
        r"""The ProcessContext of this matrix."""
        def __get__(self):
            return self._context


    property dtype:
        r"""The datatype of this matrix."""
        def __get__(self):
            return self._dtype


    def __init__(self, globalsize, dtype=np.float64, blocksize=None, context=None):
        r"""Initialise an empty DistributedMatrix.

        Parameters
        ----------
        globalsize : list of integers
            The size of the global matrix eg. [Nr, Nc].
        dtype : np.dtype, optional
            The datatype of the array. Only `float32`, `float64` (default) and
            `complex128` are supported by Scalapack.
        blocksize: list of integers, optional
            The blocking size, packed as [Br, Bc]. If `None` uses the default blocking
            (set via `initmpi`).
        context : ProcessContext, optional
            The process context. If not set uses the default (recommended). 
        """

        ## Check and set data type
        dtypes = [np.float32, np.float64, np.complex128]

        if dtype not in dtypes:
            raise Exception("Requested dtype not supported by Scalapack.")

        self._dtype = np.dtype(dtype)

        ## Check and set globalsize, and blocksize
        self.Nr, self.Nc = globalsize
        if not _blocksize and not blocksize:
            raise Exception("No supplied or default blocksize.")

        self.Br, self.Bc = blocksize if blocksize else _blocksize
            
        if not context and  not _context:
            raise Exception("No supplied or default context.")
        self._context = context if context else _context

        self._local_array = np.zeros(self.local_shape(), order='F', dtype=dtype)

        self._mkdesc()

        
    @classmethod
    def fromfile(cls, file, globalsize, dtype, blocksize=None, context=None):
        r"""Create a DistributedMatrix from a file representing the
        global matrix.

        As the file is read in via mmap, the individual blocks must be
        page aligned.

        Parameters
        ----------
        file : string
            Name of file to read.
        globalsize : list of integers
            The size of the global matrix, as [Nr, Nc].
        dtype : np.dtype
            The datatype of the array. Only `float32`, `float64` and
            `complex128` are supported by Scalapack.
        blocksize: list of integers, optional
            The blocking size as [Br, Bc]. If `None` uses the default blocking
            (set via `initmpi`).
        context : ProcessContext, optional
            The process context. If not set uses the default (recommended).

        Returns
        -------
        dm : DistributedMatrix
        """
        m = cls(globalsize, blocksize=blocksize, dtype=dtype, context=context)
        
        if os.path.exists(file):
            m._loadfile(file)

        return m


    @classmethod
    def fromarray(cls, array, blocksize=None, context=None):

        r"""Create a DistributedMatrix directly from the global `array`.

        Parameters
        ----------
        array : ndarray
            The global array to extract the local segments of.
        blocksize: list of integers, optional
            The blocking size in [Br, Bc]. If `None` uses the default
            blocking (set via `initmpi`).
        context : ProcessContext, optional
            The process context. If not set uses the default (recommended).
            
        Returns
        -------
        dm : DistributedMatrix
        """
        if array.ndim != 2:
            raise Exception("Array must be 2d.")

        m = cls(array.shape, blocksize=blocksize, dtype=array.dtype, context=context)
        
        ac = np.asfortranarray(array)
        m._loadarray(ac)

        return m


    @classmethod
    def empty_like(cls, mat):
        r"""Create a DistributedMatrix, with the same shape and
        blocking as `mat`.

        Parameters
        ----------
        mat : DistributedMatrix
            The matrix to copy.

        Returns
        -------
        cmat : DistributedMatrix
        """
        return cls([mat.Nr, mat.Nc], blocksize=[mat.Br, mat.Bc], dtype=mat.dtype, context=mat.context)
    
    
    @classmethod
    def from_npy(cls, fname, blocksize=None, shape_override=None,
                 order_override=None, context=None):
        r"""Create a distributed matrix by reading a .npy file.

        Parameters
        ----------
        fname : str
            File name to read.
        comm : mpi communicator
            Does nothing for now, but will eventually specify the communicator.
        blocksize : list of integers, optional
            The blocking size in [Br, Bc]. If `None` uses the default
            blocking (set via `initmpi`).
        shape_override : tuple of integers or None
            If not None, ignore the array shape stored in the .npy header and
            use this instead.  This is like a reshape on read operation.
        order_override : "C", "F" or None
            If not None, ignore the axis ordering specified in the .npy header
            and use the ordering specified by this parameter. This might be a
            good idea if your matrix is symetric and C ordered and you'dd
            rather read it as Fortran ordered which should be faster.
        context : ProcessContext, optional
            The process context. If not set uses the default (recommended).
        """
        
        shape, fortran_order, dtype, offset = npyutils.read_header_data(fname)

        # Global shape.
        if shape_override:
            shape = shape_override
        if len(shape) != 2:
            msg = "Distributed matrices must be 2D arrays"
            raise ValueError(msg)
        # Axis ordering.
        if order_override == 'C':
            fortran_order = False
        elif order_override == 'F':
            fortran_order = True
        if fortran_order:
            order = 'F'
        else:
            order = 'C'
        # Block size.
        if not blocksize:
            blocksize = _blocksize
        
        # Check the file size.
        file_size = os.path.getsize(fname)  # bytes.
        array_size = 1
        for s in shape:
            array_size *= s
        if file_size < np.dtype(dtype).itemsize * array_size + offset:
            raise RuntimeError("File isn't big enough")

        m = cls(shape, blocksize=blocksize, dtype=np.dtype(dtype), context=context)
        # The returned matrix is C ordered if order = 'C' but the assignment
        # always makes the local array fortran ordered.
        m.local_array[...] = blockcyclic.mpi_readmatrix(fname, m.context.mpi_comm,
                                shape, np.dtype(dtype).type, blocksize, 
                                (m.context.num_rows, m.context.num_cols),
                                order=order, displacement=offset)
        return m


    def local_shape(self):
        r"""The shape of the local matrix segment.
        
        Returns
        -------
        shape : list of integers
            The shape as [rows, cols].
        """
        nr = numrc(self.Nr, self.Br, self.context.row, 0, self.context.num_rows)
        nc = numrc(self.Nc, self.Bc, self.context.col, 0, self.context.num_cols)

        return (nr, nc)


    def _mkdesc(self):
        self._desc = np.zeros(9, dtype=np.int32)

        self._desc[0] = 1 # Dense matrix
        self._desc[1] = self.context.blacs_context
        self._desc[2] = self.Nr
        self._desc[3] = self.Nc
        self._desc[4] = self.Br
        self._desc[5] = self.Bc
        self._desc[6] = 0
        self._desc[7] = 0
        self._desc[8] = self.local_shape()[0]


    cdef int * _getdesc(self):
        return <int *>self._desc.data


    cdef char * _data(self):
         return <char *>self._local_array.data


    def _loadfile(self, file):
        bc2d_mmap_load(file, self._data(), self.dtype.itemsize, self.Nr, self.Nc, self.Br, self.Bc, 
                       self.context.num_rows, self.context.num_cols, 
                       self.context.row, self.context.col)


    def _loadarray(self, array):
        if array.dtype != self.dtype:
            raise Exception("Incompatible data types.")
        
        bc2d_copy_forward(<char *>np_data(array), <char *>self._data(),
                          self.dtype.itemsize, self.Nr, self.Nc, self.Br, self.Bc,
                          self.context.num_rows, self.context.num_cols,
                          self.context.row, self.context.col)
    

    def tofile(self, fname):
        r"""Save the distributed matrix out to a file.
        
        This use mmap to write out the local sections of the global
        matrix. The file will have the global matrix in the canonical
        order, though each block will be page aligned.

        In order to avoid a race condition, only the rank-0 MPI
        process will check for the existence of the file, and ensure
        it is long enough to save the vector to.
        
        Parameters
        ----------
        fname : string
            Name of the file to write into.
        """
         
        if self.context.mpi_rank == 0:
            length = num_rpage(self.Nr, self.Br, self.dtype.itemsize) * self.Nc * self.dtype.itemsize
            ensure_filelength(fname, length)

        self.context.mpi_comm.Barrier()

        bc2d_mmap_save(fname, <char *>self._data(), self.dtype.itemsize,
                       self.Nr, self.Nc, self.Br, self.Bc, 
                       self.context.num_rows, self.context.num_cols, 
                       self.context.row, self.context.col)


    def to_npy(self, fname, fortran_order=True, shape_override=None):
        r"""same the distributed matrix out to a .npy file.

        Parameters
        ----------
        fname : str
            File name to save to.
        fortran_order : boolian
            Order the matrix in Fortran order on disk as opposed to C order.
        shape_override : tuple of integers
            What shape to write to the header of the .npy file.  This is like a
            reshape on write operaton.
        """

        # Global shape.
        if shape_override:
            size = 1
            for s in shape_override:
                size *= s
            # Might consider relaxing this restriction.
            if size != self.Nr * self.Nc:
                msg = "Total size of the array cannot change."
                raise ValueError(msg)
            shape = shape_override
        else:
            size = self.Nr * self.Nc
            shape = (self.Nr, self.Nc)
        # Axis ordering.
        if fortran_order:
            order = 'F'
            arr = self.local_array
        else:
            order = 'C'
            arr = np.ascontiguousarray(self.local_array)

        header_data = npyutils.pack_header_data(shape, fortran_order, 
                                               self._dtype)
        header_len = npyutils.get_header_length(header_data)
        
        blockcyclic.mpi_writematrix(fname, arr, self.context.mpi_comm, 
                        (self.Nr, self.Nc), self._dtype.type,
                        (self.Br, self.Bc), 
                        (self.context.num_rows, self.context.num_cols),
                        order=order, displacement=header_len)
 
        # Write the header data.
        if self.context.mpi_rank == 0:
            npyutils.write_header_data(fname, header_data)

        self.context.mpi_comm.Barrier()


    def indices(self, full=False):
        r"""The indices of the elements stored in the local matrix.

        This can be used to easily build up distributed matrices that
        depend on their co-ordinates.

        Parameters
        ----------
        full : boolean, optional
            If False (default), the matrices of indices are not
            fleshed out, if True the full matrices are returned. This
            is like the difference between np.ogrid and np.mgrid.

        Returns
        -------
        im : tuple of ndarrays
            The first element contains the matrix of row indices and
            the second of column indices.

        Notes
        -----

        As an example a DistributedMatrix defined globally as
        :math:`M_{ij} = i + j` can be created by:

        >>> dm = DistributedMatrix(100, 100)
        >>> rows, cols = dm.indices()
        >>> dm.local_array[:] = rows + cols

        """

        if full:
            lr, lc = self.local_shape()
            ri, ci = self.indices(full=False)
            ri = np.tile(ri, lc)
            rc = np.tile(ci.T, lr).T
        else:
            ri = index_array(self.Nr, self.Br, self.context.row, self.context.num_rows)[:,np.newaxis]
            ci = index_array(self.Nc, self.Bc, self.context.col, self.context.num_cols)[np.newaxis,:]

        return (ri, ci)


    def copy(self):
        r"""Make a deep copy of the matrix.

        Returns
        -------
        c : DistributedMatrix
            A copy of this DistributedMatrix.
        """
        c = DistributedMatrix.empty_like(self)
        c.local_array[:] = self.local_array

        return c




def matrix_equal(A, B):
    r"""Test if two matrices are identical.

    Does a comparison of the global matrices. Uses MPI to test
    equality of all segments, and returns the global result for all.

    Parameters
    ----------
    A, B : DistributedMatrix
        Matrices to compare

    Returns
    -------
    cmp : boolean
        Whether equal or not.
    """
    if not np.array_equal(A.desc, B.desc):
        raise Exception("Matrices must be the same size, and equally distributed, for comparison.")

    t = np.array(np.array_equal(A.local_array, B.local_array), dtype=np.uint8)

    tv = np.zeros(A.context.mpi_size, dtype=np.uint8)

    A.context.mpi_comm.Allgather([t, MPI.BYTE], [tv, MPI.BYTE])

    return tv.astype(np.bool).all()

