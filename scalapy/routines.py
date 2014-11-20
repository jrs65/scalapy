"""
==============================================
Highlevel Interface (:mod:`~scalapy.routines`)
==============================================

This module presents a high-level interface to ScaLAPACK modelled on the API
of ``scipy.linalg``. Presently this is quite limited, and is growing as
required.

Routines
========

.. autosummary::
    :toctree: generated/

    eigh
    cholesky
    dot

"""

import numpy as np

from . import core, util
from . import lowlevel as ll


def eigh(A, B=None, lower=True, overwrite_a=True, overwrite_b=True, type_=1, eigvals=None):
    """Find the eigen-solution of a symmetric/hermitian matrix.

    Use ScaLAPACK to compute the eigenvalues and eigenvectors of a distributed
    matrix. This routine can also solve the generalised eigenvalue problem.

    Parameters
    ----------
    A : DistributedMatrix
        A complex hermitian, or real symmetric matrix to eigensolve.
    B : DistributedMatrix, optional
        A complex hermitian, or real symmetric positive definite matrix.
    lower : boolean, optional
        Scalapack uses only half of the matrix, by default the lower
        triangle will be used. Set to False to use the upper triangle.
    overwrite_a, overwrite_b : boolean, optional
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

    A = A if overwrite_a else A.copy()

    if B is None:
        # Solve the standard eigenvalue problem.

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
        m, nz, info = func(*args)

        if info < 0:
            raise Exception("Failure.")

        return evals, evecs

    else:
        # Otherwise we need to solve the generalised eigenvalue problem
        B = B if overwrite_b else B.copy()

        # Validate type
        if type_ not in [1, 2, 3]:
            raise core.ScalapackException("Type argument invalid.")

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
        evals = np.zeros(N, dtype=util.real_equiv(A.dtype))

        # Construct the arguments list for the first part
        args1 = [type_, task, erange, uplo, N, A, B, 1.0, 1.0, low, high, 0.0, evals, -1.0, evecs]

        # Construct the second half of the arguments list, these are mostly
        # the useless 'expert' mode arguments
        npmul = np.prod(A.context.grid_shape)  # NPROW * NPCOL
        ifail = np.zeros(N, dtype=np.int32)
        iclustr = np.zeros(2 * npmul, dtype=np.float64)  # Weird issue in f2py wants this to be float64?
        gap = np.zeros(npmul, dtype=util.real_equiv(A.dtype))

        args2 = [ifail, iclustr, gap]

        call_table = {'S': (ll.pssygvx, args1 + [ll.WorkArray('S', 'I')] + args2),
                      'D': (ll.pdsygvx, args1 + [ll.WorkArray('D', 'I')] + args2),
                      'C': (ll.pchegvx, args1 + [ll.WorkArray('C', 'S', 'I')] + args2),
                      'Z': (ll.pzhegvx, args1 + [ll.WorkArray('Z', 'D', 'I')] + args2)}

        func, args = call_table[A.sc_dtype]
        m, nz, info = func(*args)

        if info < 0:
            raise Exception("Failure.")

        return evals, evecs


def cholesky(A, lower=False, overwrite_a=False, zero_triangle=True):
    """Compute the Cholesky decomposition of a symmetric/hermitian matrix.

    Parameters
    ----------
    A : DistributedMatrix
        The matrix to decompose.
    lower : boolean, optional
        Compute the upper or lower Cholesky factor. Additionally Scalapack
        will only touch the upper or lower triangle of A.
    overwrite_a : boolean, optional
        By default the input matrix is destroyed, if set to False a
        copy is taken and operated on.
    zero_triangle : boolean, optional
        By default Scalapack ignores the other triangle, if set, we explicitly
        zero it.

    Returns
    -------
    cholesky : DistributedMatrix
        The Cholesky factor as a DistributedMatrix.
    """

    # Check if matrix is square
    util.assert_square(A)

    A = A if overwrite_a else A.copy()

    uplo = "L" if lower else "U"
    N = A.global_shape[0]

    args = [uplo, N, A]

    call_table = {'S': (ll.pspotrf, args),
                  'D': (ll.pdpotrf, args),
                  'C': (ll.pcpotrf, args),
                  'Z': (ll.pzpotrf, args)}

    func, args = call_table[A.sc_dtype]
    info = func(*args)

    if info < 0:
        raise core.ScalapackException("Failure.")

    ## Zero other triangle
    # by default scalapack doesn't touch the other triangle
    # (determined by upper arg). We explicitly zero it here.
    if zero_triangle:
        ri, ci = A.indices()

        # Create a mask of the other triangle
        mask = (ci <= ri) if lower else (ci >= ri)
        A.local_array[:] = A.local_array * mask

    return A


def dot(A, B, transA='N', transB='N'):
    """Parallel matrix multiplication.

    Parameters
    ----------
    A, B : DistributedMatrix
        Matrices to multiply.
    transA, transB : ['N', 'T', 'C']
        Whether we should use a transpose, rather than A or B themselves.
        Either, do nothing ('N'), normal transpose ('T'), or Hermitian transpose ('C').

    Returns
    -------
    C : DistributedMatrix
    """

    if transA not in ['N', 'T', 'C']:
        raise core.ScalapyException("Trans argument for matrix A invalid")
    if transB not in ['N', 'T', 'C']:
        raise core.ScalapyException("Trans argument for matrix B invalid")
    if A.dtype != B.dtype:
        raise core.ScalapyException("Matrices must have same type")
    # Probably should validate context too

    m = A.global_shape[0] if transA == 'N' else A.global_shape[1]
    n = B.global_shape[1] if transB == 'N' else B.global_shape[0]
    k = A.global_shape[1] if transA == 'N' else A.global_shape[0]
    l = B.global_shape[0] if transB == 'N' else B.global_shape[1]

    if l != k:
        raise core.ScalapyException("Matrix shapes are incompatible.")

    C = core.DistributedMatrix([m, n], dtype=A.dtype, block_shape=A.block_shape, context=A.context)

    args = [transA, transB, m, n, k, 1.0, A, B, 0.0, C]

    call_table = { 'S': (ll.psgemm, args),
                   'C': (ll.pcgemm, args),
                   'D': (ll.pdgemm, args),
                   'Z': (ll.pzgemm, args) }


    func, args = call_table[A.sc_dtype]
    func(*args)

    return C


def lu(A, overwrite_a=True):
    """Computes the LU factorization of a general m-by-n distributed matrix.

    The decomposition is::

        A = P * L * U

    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n) and U is upper
    triangular (upper trapezoidal if m < n). L and U are stored in A.

    Parameters
    ----------
    A : DistributedMatrix
        The matrix to decompose.
    overwrite_a : boolean, optional
        By default the input matrix is destroyed, if set to False a
        copy is taken and operated on.

    Returns
    -------
    A : DistributedMatrix
        Overwritten by local pieces of the factors L and U from the
        factorization A = P*L*U. The unit diagonal elements of L are
        not stored.
    ipiv : np.ndarray
        Array contains the pivoting information: local row i was
        interchanged with global row ipiv[i]. This array is tied to
        the distributed matrix A.

    """

    A = A if overwrite_a else A.copy()

    M, N = A.global_shape

    ipiv = np.zeros((A.local_shape[0] + A.block_shape[0]), dtype='i')

    args = [M, N, A, ipiv]

    call_table = {'S': (ll.psgetrf, args),
                  'D': (ll.pdgetrf, args),
                  'C': (ll.pcgetrf, args),
                  'Z': (ll.pzgetrf, args)}

    func, args = call_table[A.sc_dtype]
    info = func(*args)

    if info < 0:
        raise core.ScalapackException("Failure.")

    return A, ipiv


def inv(A, overwrite_a=True):
    """Computes the inverse of a LU-factored distributed matrix.

    Computes the inverse of a general distributed matrix A using the
    LU factorization. This method inverts U and then computes the
inverse
    of A by solving the system inv(A)*L = inv(U) for inv(A).

    Parameters
    ----------
    A : DistributedMatrix
        The matrix to inverse.
    overwrite_a : boolean, optional
        By default the input matrix is destroyed, if set to False a
        copy is taken and operated on.

    Returns
    -------
    inv : DistributedMatrix
        The inverse of `A`.
    ipiv : np.ndarray
        Array contains the pivoting information. If ipiv[i]=j, then
        the local row i was swapped with the global row j.
        This array is tied to the distributed matrix A.

    """

    # Check if matrix is square
    util.assert_square(A)

    A = A if overwrite_a else A.copy()

    N, N = A.global_shape

    # first do the LU factorization
    ipiv = np.zeros((A.local_shape[0] + A.block_shape[0]), dtype='i')

    args = [N, N, A, ipiv]

    call_table = {'S': (ll.psgetrf, args),
                  'D': (ll.pdgetrf, args),
                  'C': (ll.pcgetrf, args),
                  'Z': (ll.pzgetrf, args)}

    func, args = call_table[A.sc_dtype]
    info = func(*args)

    if info < 0:
        raise core.ScalapackException("Failure.")

    # then computes the inverse of a LU-factored distributed matrix.
    args = [N, A, ipiv]

    call_table = {'S': (ll.psgetri, args + [ll.WorkArray('S', 'I')]),
                  'D': (ll.pdgetri, args + [ll.WorkArray('D', 'I')]),
                  'C': (ll.pcgetri, args + [ll.WorkArray('C', 'I')]),
                  'Z': (ll.pzgetri, args + [ll.WorkArray('Z', 'I')])}

    func, args = call_table[A.sc_dtype]
    info = func(*args)

    if info < 0:
        raise core.ScalapackException("Failure.")

    return A, ipiv


def pinv(A, overwrite_a=True):
    """ Compute the (Moore-Penrose) pseudo-inverse of a distributed matrix.

    Calculate a generalized inverse of a distributed matrix using a
    least-squares solver.

    NOTE: To get correct answer, `A` must have full rank.

    Parameters
    ----------
    A : DistributedMatrix
        Matrix to be pseudo-inverted.
    overwrite_a : boolean, optional
        By default the input matrix is destroyed, if set to False a
        copy is taken and operated on.

    Returns
    -------
    pinv : DistributedMatrix
        The pseudo-inverse of matrix `A` is contained in the fist n rows of pinv for the distributed matrix of global shape (m, n).

    """

    # Check if matrix is square
    # util.assert_square(A)

    A = A if overwrite_a else A.copy()

    M, N = A.global_shape

    # distributed matrix which contains an identity matrix in the first M rows
    B = core.DistributedMatrix([max(M, N), M], dtype=A.dtype, block_shape=A.block_shape, context=A.context)
    (g,r,c) = B.local_diagonal_indices(allow_non_square=True)
    B.local_array[r,c] = 1.0

    args = ['N', M, N, M, A, B]

    call_table = {'S': (ll.psgels, args + [ll.WorkArray('S')]),
                  'D': (ll.pdgels, args + [ll.WorkArray('D')]),
                  'C': (ll.pcgels, args + [ll.WorkArray('C')]),
                  'Z': (ll.pzgels, args + [ll.WorkArray('Z')])}

    func, args = call_table[A.sc_dtype]
    info = func(*args)

    if info < 0:
        raise core.ScalapackException("Failure.")

    return B


def transpose(A):
    """Transpose the distributed matrix.

    Parameters
    ----------
    A : DistributedMatrix
        The matrix to transpose.

    Returns
    -------
    trans : DistributedMatrix
        The transpose of `A`

    """

    return A.transpose()


def conj(A):
    """Complex conjugate a distributed matrix

    Parameters
    ----------
    A : DistributedMatrix
        The matrix to complex conjugate.

    Returns
    -------
    cj : DistributedMatrix
        The complex conjugate of `A`

    """

    return A.conj()


def hconj(A):
    """Hermitian conjugate a distributed matrix, i.e., transpose and
    complex conjugate the distributed matrix.

    Parameters
    ----------
    A : DistributedMatrix
        The matrix to Hermitian conjugate.

    Returns
    -------
    hermi : DistributedMatrix
        The Hermitian conjugate of `A`

    """

    return A.hconj()
