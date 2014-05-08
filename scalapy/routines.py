import numpy as np

from . import core, util
from . import lowlevel as ll



def eigh(A, lower=True, overwrite=True, eigvals=None):
    """Compute the eigen-decomposition of a symmetric/hermitian matrix.

    Use Scalapack to compute the eigenvalues and eigenvectors of a
    distributed matrix.

    Parameters
    ----------
    mat : DistributedMatrix
        The matrix to decompose.
    lower : boolean, optional
        Scalapack uses only half of the matrix, by default the lower
        triangle will be used. Set to False to use the upper triangle.
    overwrite : boolean, optional
        By default the input matrix is destroyed, if set to False a
        copy is taken and operated on.
    eigvals : tuple (lo, hi), optional    
        Indices of the lowest and highest eigenvalues you would like to
        calculate. Indexed from zero.

    Returns
    -------
    evals : np.ndarray
        The eigenvalues of the matrix, they are returned as a global
        numpy array of all values.
    evecs : DistributedMatrix
        The eigenvectors as a DistributedMatrix.
    """

    # Check if matrix is square
    util.assert_square(A)

    A = A if overwrite else A.copy()

    task = 'V'
    erange = 'A'
    uplo = "L" if lower else "U"
    N = A.global_shape[0]
    low, high = 1, 1

    # Get eigval indices if set
    if eigvals is not None:
        low = eigvals[0] + 1
        high = eigvals[1] + 1
        erange = 'I'

    evecs = core.DistributedMatrix.empty_like(A)
    evals = np.empty(N, dtype=util.real_equiv(A.dtype))

    args = [task, erange, uplo, N, A, 1.0, 1.0, low, high, evals, evecs]

    call_table = {'S': (ll.pssyevr, args + [ll.WorkArray('S', 'I')]),
                  'D': (ll.pdsyevr, args + [ll.WorkArray('D', 'I')]),
                  'C': (ll.pcheevr, args + [ll.WorkArray('C', 'S', 'I')]),
                  'Z': (ll.pzheevr, args + [ll.WorkArray('Z', 'D', 'I')])}

    func, args = call_table[A.sc_dtype]
    info, m, nz = func(*args)


    args = [task, uplo, N, A, evals, evecs]

    # call_table = {'S': (ll.pssyevd, args + [ll.WorkArray('S', 'I')]),
    #               'D': (ll.pdsyevd, args + [ll.WorkArray('D', 'I')]),
    #               'C': (ll.pcheevd, args + [ll.WorkArray('C', 'S', 'I')]),
    #               'Z': (ll.pzheevd, args + [ll.WorkArray('Z', 'D', 'I')])}

    # func, args = call_table[A.sc_dtype]
    # info = func(*args)

    if info < 0:
        raise Exception("Failure.")

    return evals, evecs
