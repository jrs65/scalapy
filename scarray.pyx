

import numpy as np
cimport numpy as np

#from libc.stdlib import size_t

ctypedef unsigned long size_t

cdef extern from "bcutil.h":
    int bc1d_copy_pagealign(double * src, double * dest, int N, int B)
    int bc2d_copy_pagealign(double * src, double * dest, int Nr, int Nc, int Br, int Bc)
    int bc1d_from_pagealign(double * src, double *dest, size_t N, size_t B)
    int bc2d_from_pagealign(double * src, double * dest, size_t Nr, size_t Nc, size_t Br, size_t Bc)
    int num_rpage(int N, int B)


_context = None
    
class ProcessContext(object):
    r"""The position in the process grid."""
    num_rows = 1
    num_cols = 1

    row = 0
    col = 0


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

    

cdef class ScVector(object):

    cdef double * data

    local_vector = None
    global_vector = None

    def __init__(self, N, B, fname = None, context = None):
        self.N = N
        self.B = B

        if not context:
            if not _context:
                raise Exception("No supplied or default context.")
            else:
                self.context = _context
        else:
            self.context = context
        
        if fname:
            pass

    def to_file(fname):
        pass


cdef class ScMatrix(object):

    cdef double * data

    local_matrix = None
    global_matrix = None

    def __init__(self, Nr, Nc, Br, Bc, fname = None, context = None):
        pass

    def to_file(fname):
        pass
    
