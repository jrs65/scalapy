
cimport numpy as np
import numpy as np


from libc.stddef cimport size_t
from libc.stdlib cimport malloc, free

from scalapack cimport *

from scarray cimport *
#from scarray import *


cdef int _ONE = 1
cdef int _ZERO = 0


def pdsyevd(mat, destroy = True, upper = True):
    r"""Compute the eigen-decomposition of a symmetric matrix.

    Use Scalapack to compute the eigenvalues and eigenvectors of a
    distributed matrix.

    Parameters
    ----------
    mat : DistributedMatrix
        The matrix to decompose.
    destroy : boolean, optional
        By default the input matrix is destroyed, if set to False a
        copy is taken and operated on.
    upper : boolean, optional
        Scalapack uses only half of the matrix, by default the upper
        triangle will be used. Set to False to use the lower triangle.

    Returns
    -------
    evals : np.ndarray
        The eigenvalues of the matrix, they are returned as a global
        numpy array of all values.
    evecs : DistributedMatrix
        The eigenvectors as a DistributedMatrix.
    """
    cdef int lwork, liwork
    cdef double * work
    cdef double wl
    cdef int * iwork
    cdef DistributedMatrix A, evecs

    cdef int info
    cdef np.ndarray evals
    
    A = mat if destroy else mat.copy()

    evecs = DistributedMatrix.empty_like(A)
    evals = np.empty(A.Nr, dtype=np.float64)

    liwork = 7*A.Nr + 8*A.context.num_cols + 2
    iwork = <int *>malloc(sizeof(int) * liwork)

    uplo = "U" if upper else "L"

    ## Workspace size inquiry
    lwork = -1
    pdsyevd_("V", uplo, &(A.Nr),
             A._data(), &_ONE, &_ONE, A._getdesc(),
             <double *>np_data(evals),
             evecs._data(), &_ONE, &_ONE, evecs._getdesc(),
             &wl, &lwork, iwork, &liwork,
             &info);
    
    ## Initialise workspace to correct length
    lwork = <int>wl
    work = <double *>malloc(sizeof(double) * lwork)

    ## Compute eigen problem
    pdsyevd_("V", uplo, &(A.Nr),
             A._data(), &_ONE, &_ONE, A._getdesc(),
             <double *>np_data(evals),
             evecs._data(), &_ONE, &_ONE, evecs._getdesc(),
             work, &lwork, iwork, &liwork,
             &info);

    free(iwork)
    free(work)

    return (evals, evecs)
    


