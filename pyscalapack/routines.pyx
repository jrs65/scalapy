
cimport numpy as np
import numpy as np


from libc.stddef cimport size_t
from libc.stdlib cimport malloc, free

from scalapack cimport *

from pyscalapack.core cimport *
#from scarray import *


cdef int _ONE = 1
cdef int _ZERO = 0


def flatten(x):
    """flatten(sequence) -> list

    Returns a single, flat list which contains all elements retrieved
    from the sequence and all recursively contained sub-sequences
    (iterables).

    Examples:
    >>> [1, 2, [3,4], (5,6)]
    [1, 2, [3, 4], (5, 6)]
    >>> flatten([[[1,2,3], (42,None)], [4,5], [6], 7, MyVector(8,9,10)])
    [1, 2, 3, 42, None, 4, 5, 6, 7, 8, 9, 10]"""

    result = []
    for el in x:
        #if isinstance(el, (list, tuple)):
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def assert_square(A):
    Alist = flatten([A])
    for A in Alist:
        if A.Nr != A.Nc:
            raise Exception("Matrix must be square (has dimensions %i x %i)." % (A.Nr, A.Nc))

def assert_type(A, dtype):
    Alist = flatten([A])
    for A in Alist:
        if A.dtype != dtype:
            raise Exception("Expected Matrix to be of type %s, got %s." % (repr(dtype), repr(A.dtype)))
    


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

    ## Check input
    assert_type(mat, np.float64)
    assert_square(mat)
    
    A = mat if destroy else mat.copy()

    evecs = DistributedMatrix.empty_like(A)
    evals = np.empty(A.Nr, dtype=np.float64)

    liwork = 7*A.Nr + 8*A.context.num_cols + 2
    iwork = <int *>malloc(sizeof(int) * liwork)

    uplo = "U" if upper else "L"

    ## Workspace size inquiry
    lwork = -1
    pdsyevd_("V", uplo, &(A.Nr),
             <double *>A._data(), &_ONE, &_ONE, A._getdesc(),
             <double *>np_data(evals),
             <double *>evecs._data(), &_ONE, &_ONE, evecs._getdesc(),
             &wl, &lwork, iwork, &liwork,
             &info);
    
    ## Initialise workspace to correct length
    lwork = <int>wl
    work = <double *>malloc(sizeof(double) * lwork)

    ## Compute eigen problem
    pdsyevd_("V", uplo, &(A.Nr),
             <double *>A._data(), &_ONE, &_ONE, A._getdesc(),
             <double *>np_data(evals),
             <double *>evecs._data(), &_ONE, &_ONE, evecs._getdesc(),
             work, &lwork, iwork, &liwork,
             &info);

    free(iwork)
    free(work)

    return (evals, evecs)
    




def pdgemm(DistributedMatrix A, DistributedMatrix B, DistributedMatrix C = None, alpha = 1.0, beta = 1.0, transa = False, transb = False, destroyc = True):
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
    cdef int m, n, k
    cdef DistributedMatrix Cm
    cdef int info
    cdef double a, b

    assert_type([A, B, C], np.float64)
    
    a = alpha
    b = beta

    m = A.Nr if not transa else A.Nc
    k = A.Nc if not transa else A.Nr

    n = B.Nc if not transb else B.Nr
    k2 = B.Nr if not transb else B.Nc

    ## Check matrix sizes A, B, are compatible
    if k != k2:
        raise Exception("Matrices A and B have incompatible shapes for multiplication.")

    ## Ensure C has correct size, and copy if required or create if not passed in.
    if C != None:
        if m != C.Nr or n != C.Nc:
            raise Exception("Matrix C is not compatible with matrices A and B.")

        Cm = C if destroyc else C.copy()

    else:
        Cm = DistributedMatrix(globalsize = [m, n], dtype=A.dtype, blocksize = [A.Br, A.Bc], context = A._context)

    tA = "N" if not transa else "T"
    tB = "N" if not transb else "T"

    pdgemm_(tA, tB, &m, &n, &k, &a,
            <double *>A._data(), &_ONE, &_ONE, A._getdesc(),
            <double *>B._data(), &_ONE, &_ONE, B._getdesc(),
            &b,
            <double *>Cm._data(), &_ONE, &_ONE, Cm._getdesc())

    return Cm



def pdpotrf(mat, destroy = True, upper = True):
    r"""Compute the Cholesky decomposition of a symmetric positive definite
    matrix.

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
    cholesky : DistributedMatrix
        The Cholesky decomposition of the matrix.
    """
    cdef DistributedMatrix A

    cdef int info


    ## Check input
    assert_type(mat, np.float64)
    assert_square(mat)
    
    A = mat if destroy else mat.copy()

    uplo = "U" if upper else "L"
    
    pdpotrf_(uplo, &(A.Nr),
             <double *>A._data(), &_ONE, &_ONE, A._getdesc(),
             &info)

    if info < 0:
        raise Exception("Something weird has happened.")
    elif info > 0:
        raise Exception("Matrix is not positive definite.")

    ## Zero other triangle
    # by default scalapack doesn't touch the other triangle
    # (determined by upper arg). We explicitly zero it here.
    ri, ci = A.indices()
    if upper:
        A.local_array[np.where(ci - ri < 0)] = 0.0
    else:
        A.local_array[np.where(ci - ri > 0)] = 0.0
        
    return A


