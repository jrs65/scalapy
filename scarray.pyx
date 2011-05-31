import os
import os.path
import sys
import numpy as np
cimport numpy as np
import sys

from mpi4py import MPI


#ctypedef unsigned long size_t
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

    
    cdef int pnum, nprocs, ictxt, row, col, nrows, ncols
    cdef int rank, size

    global _context
    global _blocksize

    # MPI setup
    comm = MPI.COMM_WORLD
    ct = ProcessContext()
    ct.mpi_rank = comm.Get_rank()
    ct.mpi_size = comm.Get_size()
    #print "MPI: %i of %i" % (ct.mpi_rank, ct.mpi_size)
    
    Cblacs_pinfo(&pnum, &nprocs)
    #print "BLACS pinfo %i %i" % (pnum, nprocs)

    ## Figure out what to do when we have spare MPI procs
    if not gridsize:
        side = int((nprocs*1.0)**0.5)
        gridsize = [side, side]

    gs = gridsize[0]*gridsize[1]
    if gs > ct.mpi_size:
        raise Exception("Requested gridsize is larger than number of MPI processes.")
    if gs < ct.mpi_size:
        import warnings
        warnings.warn("More MPI processes than process grid points. This may go crazy (especially for commparisons).")

    # Initialise BLACS process grid
    Cblacs_get(-1, 0, &ictxt)
    ct.blacs_context = ictxt
    #print "BLACS context: %i" % ictxt
    
    Cblacs_gridinit(&ictxt, "Row", gridsize[0], gridsize[1])
    Cblacs_gridinfo(ictxt, &nrows, &ncols, &row, &col)

    # Fill out default ProcessContext
    ct.num_rows = nrows
    ct.num_cols = ncols

    ct.row = row
    ct.col = col
    #print "MPI %i: position (%i,%i) in %i x %i" % (ct.mpi_rank, ct.row, ct.col, ct.num_rows, ct.num_cols)

    _context = ct

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

    mpi_rank = 0
    mpi_size = 1

    blacs_context = 0


def vector_pagealign(vec, blocksize):
    r"""Page aligns the blocks in a matrix, and makes Fortran ordered.

    Parameters
    ==========
    vec : ndarray
        The vector to page align (can either by C or Fortran ordered).
    blocksize : scalar
        The blocksize of the vector.

    Returns
    =======
    v2 : ndarray
        The page aligned vector.
    """
    cdef np.ndarray[np.float64_t, ndim=1] v1
    cdef np.ndarray[np.float64_t, ndim=1] v2

    v1 = np.ascontiguousarray(vec)

    N = vec.shape[0]
    B = blocksize

    nr = num_rpage(N, B)

    v2 = np.empty(nr, order='F')

    bc1d_copy_pagealign(<double *>v1.data, <double *>v2.data, N, B)

    return v2


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
    cdef np.ndarray[np.float64_t, ndim=2] m1
    cdef np.ndarray[np.float64_t, ndim=2] m2

    m1 = np.asfortranarray(mat)

    Nr, Nc = mat.shape
    Br, Bc = blocksize

    nr = num_rpage(Nr, Br)

    m2 = np.empty((nr, Nc), order='F')

    bc2d_copy_pagealign(<double *>m1.data, <double *>m2.data, Nr, Nc, Br, Bc)

    return m2


def vector_from_pagealign(vecp, size, blocksize):
    r"""Page aligns the blocks in a matrix, and makes Fortran ordered.

    Parameters
    ==========
    vecp : ndarray
        The vector to page align (can either by C or Fortran ordered).
    blocksize : scalar
        The blocksize of the vector.

    Returns
    =======
    v2 : ndarray
        The page aligned vector.
    """
    cdef np.ndarray[np.float64_t, ndim=1] v1
    cdef np.ndarray[np.float64_t, ndim=1] v2

    v1 = vecp

    N = size
    B = blocksize

    nr = num_rpage(N, B)

    if len(v1) < nr:
        raise Exception("Source vector not long enough.")

    v2 = np.empty(N, order='F')

    bc1d_from_pagealign(<double *>v1.data, <double *>v2.data, N, B)

    return v2


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
    cdef np.ndarray[np.float64_t, ndim=1] m1
    cdef np.ndarray[np.float64_t, ndim=2] m2

    #m1 = matp.flatten()
    m1 = matp.reshape(-1, order='A')

    Nr, Nc = size
    Br, Bc = blocksize

    nr = num_rpage(Nr, Br)

    if np.size(m1) < nr*Nc:
        raise Exception("Source matrix not long enough. Is %i x %i, should be %i x %i." % (matp.shape[0], matp.shape[1], nr, Nc))

    m2 = np.empty((Nr, Nc), order='F')

    bc2d_from_pagealign(<double *>m1.data, <double *>m2.data, Nr, Nc, Br, Bc)

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
    




cdef void * np_data(np.ndarray a):
    return a.data



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
        raise Exception("Error opening file: %s" % fname)

    res = posix_fallocate(fd, 0, <size_t>length)
    close(fd)
    
    if res != 0:
        raise Exception("Allocating file %s length %i bytes failed." % (fname, length))

    st = os.stat(fname)
    if st.st_size < length:
        raise Exception("Allocating file %s length %i bytes failed in some odd way.")
    

    

cdef class DistributedVector(object):
    r"""A vector distributed over multiple MPI processes.

    Attributes
    ----------
    N : integer, readonly
        The global vector length
    B : integer, readonly
        The block length
    
    """

    property local_vector:
        r"""The local, block-cyclic packed segment of the vector.
        
        This is an ndarray and is readonly. However, only the
        reference is readonly, the array itself can be modified in
        place.
        """
        def __get__(self):
            return self._local_vector
        
    property desc:
        r"""The Scalapack array descriptor. See [1]_. Returned as an integer
        ndarray and is readonly.
        
        .. [1] http://www.netlib.org/scalapack/slug/node77.html
        """
        def __get__(self):
            return self._desc.copy()
    
    property context:
        r"""The ProcessContext of this vector."""
        
        def __get__(self):
            return self._context

    def __init__(self, globalsize, blocksize = None, context = None):
        r"""Initialise an empty DistributedVector.

        Parameters
        ----------
        globalsize : integer
            The size of the global vector.
        blocksize: integer, optional
            The blocking size. If `None` uses the default row blocking
            (set via `initmpi`).
        context : ProcessContext, optional
            The process context. If not set uses the default (recommended). 
        """
        self.N = globalsize

        if not _blocksize and not blocksize:
            raise Exception("No supplied or default blocksize.")

        self.B = blocksize if blocksize else _blocksize[0]
        
        if not context and  not _context:
            raise Exception("No supplied or default context.")
        self.context = context if context else _context
        
        self.local_vector = np.empty(self.local_shape(), dtype=np.float64)

        self._mkdesc()

    def _mkdesc(self):
        ## Fill out the Scalapack array descriptor
        self._desc = np.zeros(9, dtype=np.int32)

        self._desc[0] = 1 # Dense matrix
        self._desc[1] = self.context.blacs_context
        self._desc[2] = self.N
        self._desc[3] = 1
        self._desc[4] = self.B
        self._desc[5] = 1
        self._desc[6] = 0
        self._desc[7] = 0
        self._desc[8] = self.local_shape()
    
    @classmethod
    def empty_like(cls, vec):
        r"""Create a DistributedVector, with the same shape and
        blocking as `vec`.

        Parameters
        ----------
        vec : DistributedVector
            The vector to copy.

        Returns
        -------
        cvec : DistributedVector
        """
        return cls(vec.N, vec.B)
        
    def local_shape(self):
        r"""The length of the local vector segment.

        Returns
        -------
        nr : integer
        """
        nr = numrc(self.Nr, self.Br, self.context.row, 0, self.context.num_rows)
        return nr
    
    cdef int * _getdesc(self):
        return <int *>self._desc.data
    
    cdef double * _data(self):
         return <double *>self._local_vector.data
     
    def _loadfile(self, file):
        ## Use mmap to load the file.
        bc1d_mmap_load(file, <double *>self._data(), self.N, self.B,
                       self.context.num_rows, self.context.row)
        
    def _loadarray(self, array):
        bc1d_copy_forward(<double *>np_data(array), <double *>self._data(), self.N, self.B, 
                          self.context.num_rows, self.context.row)
        
    @classmethod
    def fromfile(cls, file, globalsize, blocksize = None):
        
        r"""Create a DistributedVector from a file representing the
        global vector.

        As the file is read in via mmap, the individual blocks must be
        page aligned.

        Parameters
        ----------
        globalsize : integer
            The size of the global vector.
        blocksize: integer, optional
            The blocking size. If `None` uses the default row blocking
            (set via `initmpi`).
        context : ProcessContext, optional
            The process context. If not set uses the default (recommended).

        Returns
        -------
        dv : DistributedVector
        """
        
        v = cls(globalsize, blocksize)
        
        if os.path.exists(file):
            v._loadfile(file)

        return v

    @classmethod
    def fromarray(cls, array, blocksize = None):

        r"""Create a DistributedVector directly from the global `array`.

        As the file is read in via mmap, the individual blocks must be
        page aligned.

        Parameters
        ----------
        array : ndarray
            The global array to extract the local segments of.
        blocksize: integer, optional
            The blocking size. If `None` uses the default row blocking
            (set via `initmpi`).

        Returns
        -------
        dv : DistributedVector
        """
        if array.ndim != 1:
            raise Exception("Array must be 2d.")

        v = cls(array.shape, blocksize)
        
        ac = np.asfortranarray(array)
        v._loadarray(ac)

        return v

    
    def tofile(self, fname):
        r"""Save the distributed vector out to a file.

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
            length = num_rpage(self.N, self.B) * sizeof(double)

        ensure_filelength(fname, length)
        
        MPI.COMM_WORLD.barrier()

        bc1d_mmap_save(fname, <double *>self._data(), self.N, self.B,
                       self.context.num_rows, self.context.row) 








cdef class DistributedMatrix(object):
    r"""A matrix distributed over multiple MPI processes.

    Attributes
    ----------
    Nr, Nc : integer, readonly
        The number of rows and cokumns of the global matrix.
    Br, Bc : integer, readonly
        The block shape.
    
    """

    property local_matrix:
        r"""The local, block-cyclic packed segment of the matrix.

        This is an ndarray and is readonly. However, only the
        reference is readonly, the array itself can be modified in
        place.
        """
        def __get__(self):
            return self._local_matrix

    
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

    def __init__(self, globalsize, blocksize = None, context = None):
        r"""Initialise an empty DistributedMatrix.

        Parameters
        ----------
        globalsize : list of integers
            The size of the global matrix eg. [Nr, Nc].
        blocksize: list of integers, optional
            The blocking size, packed as [Br, Bc]. If `None` uses the default blocking
            (set via `initmpi`).
        context : ProcessContext, optional
            The process context. If not set uses the default (recommended). 
        """
        self.Nr, self.Nc = globalsize
        if not _blocksize and not blocksize:
            raise Exception("No supplied or default blocksize.")

        self.Br, self.Bc = blocksize if blocksize else _blocksize
            
        if not context and  not _context:
            raise Exception("No supplied or default context.")
        self._context = context if context else _context

        self._local_matrix = np.empty(self.local_shape(), order='F', dtype=np.float64)

        self._mkdesc()


        
    @classmethod
    def fromfile(cls, file, globalsize, blocksize = None):
        r"""Create a DistributedMatrix from a file representing the
        global matrix.

        As the file is read in via mmap, the individual blocks must be
        page aligned.

        Parameters
        ----------
        globalsize : list of integers
            The size of the global matrix, as [Nr, Nc].
        blocksize: list of integers, optional
            The blocking size as [Br, Bc]. If `None` uses the default blocking
            (set via `initmpi`).
        context : ProcessContext, optional
            The process context. If not set uses the default (recommended).

        Returns
        -------
        dm : DsitributedMatrix
        """
        m = cls(globalsize, blocksize)
        
        if os.path.exists(file):
            m._loadfile(file)

        return m

    @classmethod
    def fromarray(cls, array, blocksize = None):

        r"""Create a DistributedMatrix directly from the global `array`.

        As the file is read in via mmap, the individual blocks must be
        page aligned.

        Parameters
        ----------
        array : ndarray
            The global array to extract the local segments of.
        blocksize: list of integers, optional
            The blocking size in [Br, Bc]. If `None` uses the default
            blocking (set via `initmpi`).

        Returns
        -------
        dm : DistributedMatrix
        """
        if array.ndim != 2:
            raise Exception("Array must be 2d.")

        m = cls(array.shape, blocksize)
        
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
        return cls([mat.Nr, mat.Nc], [mat.Br, mat.Bc])
        

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


    cdef double * _data(self):
         return <double *>self._local_matrix.data


    def _loadfile(self, file):
        bc2d_mmap_load(file, <double *>self._data(), self.Nr, self.Nc, self.Br, self.Bc, 
                       self.context.num_rows, self.context.num_cols, 
                       self.context.row, self.context.col)

    def _loadarray(self, array):
        bc2d_copy_forward(<double *>np_data(array), <double *>self._data(), self.Nr, self.Nc, self.Br, self.Bc, 
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
            length = num_rpage(self.Nr, self.Br) * self.Nc * sizeof(double)
            ensure_filelength(fname, length)

        MPI.COMM_WORLD.barrier()

        bc2d_mmap_save(fname, <double *>self._data(), self.Nr, self.Nc, self.Br, self.Bc, 
                       self.context.num_rows, self.context.num_cols, 
                       self.context.row, self.context.col)

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
        >>> dm.local_matrix[:] = rows + cols

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

    t = np.array(np.array_equal(A.local_matrix, B.local_matrix), dtype=np.uint8)

    tv = np.zeros(A.context.mpi_size, dtype=np.uint8)

    MPI.COMM_WORLD.Allgather([t, MPI.BYTE], [tv, MPI.BYTE])

    return tv.astype(np.bool).all()
