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


def _pxxxevr(jobz, erange, uplo, A, vl, vu, il, iu):
    # wrapper for ScaLAPACK pssyevr, pdsyevr, pcheevr, pzheevr

    n = A.global_shape[0]

    w = np.empty(n, dtype=util.real_equiv(A.dtype))
    Z = core.DistributedMatrix.empty_like(A)

    args = [jobz, erange, uplo, n, A, vl, vu, il, iu, w, Z]

    call_table = {'S': (ll.pssyevr, args + [ll.WorkArray('S', 'I')]),
                  'D': (ll.pdsyevr, args + [ll.WorkArray('D', 'I')]),
                  'C': (ll.pcheevr, args + [ll.WorkArray('C', 'S', 'I')]),
                  'Z': (ll.pzheevr, args + [ll.WorkArray('Z', 'D', 'I')])}

    func, args = call_table[A.sc_dtype]
    m, nz, info = func(*args)

    return w, Z, m, nz, info


def _pxxxgvx(ibtype, jobz, erange, uplo, A, B, vl, vu, il, iu, abstol=0.0, orfac=-1.0):
    # wrapper for ScaLAPACK pssygvx, pdsygvx, pchegvx, pzhegvx

    N = A.global_shape[0]

    w = np.zeros(N, dtype=util.real_equiv(A.dtype))
    Z = core.DistributedMatrix.empty_like(A)

    # Construct the arguments list for the first part
    args1 = [ibtype, jobz, erange, uplo, N, A, B, vl, vu, il, iu, abstol, w, orfac, Z]

    # Construct the second half of the arguments list, these are mostly
    # the useless 'expert' mode arguments
    npmul = np.prod(A.context.grid_shape)  # NPROW * NPCOL
    ifail = np.zeros(N, dtype=np.int32)
    iclustr = np.zeros(2 * npmul, dtype=np.int32)
    gap = np.zeros(npmul, dtype=util.real_equiv(A.dtype))

    args2 = [ifail, iclustr, gap]

    call_table = {'S': (ll.pssygvx, args1 + [ll.WorkArray('S', 'I')] + args2),
                  'D': (ll.pdsygvx, args1 + [ll.WorkArray('D', 'I')] + args2),
                  'C': (ll.pchegvx, args1 + [ll.WorkArray('C', 'S', 'I')] + args2),
                  'Z': (ll.pzhegvx, args1 + [ll.WorkArray('Z', 'D', 'I')] + args2)}

    func, args = call_table[A.sc_dtype]
    m, nz, info = func(*args)

    return w, Z, m, nz, info, ifail


def eigh(A, B=None, lower=True, eigvals_only=False, overwrite_a=True, overwrite_b=True, type_=1, eigbounds=None, eigvals=None):
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
    eigvals_only : bool, optional
        Whether to calculate only eigenvalues and no eigenvectors.
        (Default: both are calculated)
    overwrite_a, overwrite_b : boolean, optional
        By default the input matrix is destroyed, if set to False a
        copy is taken and operated on.
    eigbounds : tuple (vl, vu), optional
        The lower and upper bounds of the interval to searched for eigenvalues.
        Takes precedence over `eigvals`.
    eigvals : tuple (lo, hi), optional
        Indices of the lowest and highest (inclusive) eigenvalues you would
        like to calculate. Indexed from zero, negative from the end. Take
        action only if `eigbounds` is None.

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

    task = 'N' if eigvals_only else 'V'
    erange = 'A'
    uplo = "L" if lower else "U"
    N = A.global_shape[0]
    vl, vu = 0.0, 1.0
    il, iu = 1, 1

    # Get eigval bounds if set
    if eigbounds is not None:
        vl, vu = eigbounds
        assert vl <= vu, 'Invalid range [%f, %f]' % (vl, vu)
        erange = 'V'
    # Get eigval indices if set
    elif eigvals is not None:
        il = eigvals[0] + N + 1 if eigvals[0] < 0 else eigvals[0] + 1
        iu = eigvals[1] + N + 1 if eigvals[1] < 0 else eigvals[1] + 1
        assert 1 <= il and min(il, N) <= iu <= N, 'Invalid indices %d, %d' % (il, iu)
        erange = 'I'

    A = A if overwrite_a else A.copy()

    if B is None:
        # Solve the standard eigenvalue problem.

        evals, evecs, m, nz, info = _pxxxevr(task, erange, uplo, A, vl, vu, il, iu)

        if info == 0:
            if task == 'N':
                return evals[:m]
            else:
                return evals[:m], evecs[:, :m]
        if info < 0:
            raise core.ScalapackException("Failure with info = %d" % info)
        if info > 0:
            raise core.ScalapackException("Unknown error")

    else:
        # Otherwise we need to solve the generalised eigenvalue problem

        # Check if matrix is square
        util.assert_square(B)
        assert A.global_shape == B.global_shape, 'Not the same global shape'

        B = B if overwrite_b else B.copy()

        # Validate type
        if type_ not in [1, 2, 3]:
            raise core.ScalapackException("Type argument invalid.")

        evals, evecs, m, nz, info, ifail = _pxxxgvx(type_, task, erange, uplo, A, B, vl, vu, il, iu)

        if info == 0:
            if task == 'N':
                return evals[:m]
            else:
                return evals[:m], evecs[:, :m]
        if info < 0:
            raise core.ScalapackException("Failure with info = %d" % info)
        if info > 0:
            if np.mod(info, 2) != 0:
                raise core.ScalapackException("One or more eigenvectors failed to converge")
            elif np.mod(info / 2, 2) != 0:
                raise core.ScalapackException("Eigenvectors corresponding to one or more clusters of eigenvalues could not be reorthogonalized because of insufficient workspace")
            elif np.mod(info / 4, 2) != 0:
                raise core.ScalapackException("Space limit prevented p?sygvx from computing all of the eigenvectors between %f and %f. The number of eigenvectors computed is %d" % (vl, vu, nz))
            elif np.mod(info / 8, 2) != 0:
                raise core.ScalapackException("p?stebz failed to compute eigenvalues")
            elif np.mod(info / 16, 2) != 0:
                raise core.ScalapackException("The smallest minor order %d of `B` is not positive definite" % ifail[0])
            else:
                raise core.ScalapackException("Unknown error")


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
    elif info > 0:
        raise core.ScalapackException("The leading minor of order %d is not positive-definite, and the factorization could not be completed." % info)

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


def svd(A, overwrite_a=True, compute_u=True, compute_v=True):
    """Distributed matrix Singular Value Decomposition.

    Factors the matrix `A` as U * np.diag(s) * VT, where U and VT are
    unitary and s is a 1-d array of A's singular values.

    Parameters
    ----------
    A : DistributedMatrix
        Matrix to do SVD, shape (m, n).
    overwrite_a : boolean, optional
        By default the input matrix is destroyed, if set to False a
        copy is taken and operated on.
    compute_u, compute_v : bool, optional
        Whether or not to compute U and VT in addition to s. True by default.

    Returns
    -------
    U : DistributedMatrix
        Unitary matrices with shape (m, min(m, n)). Only returned when compute_u is True.
    s : np.ndarray
        The singular values for `A` with shape (min(m, n)), sorted in descending order.
    VT : DistributedMatrix
        Unitary matrices with shape (min(m, n), n). Only returned when compute_v is True.

    """

    A = A if overwrite_a else A.copy()

    m, n = A.global_shape
    size = min(m, n)
    sizeb = max(m, n)

    # distributed matrix which contains the first size columns of U (the left singular vectors)
    U = core.DistributedMatrix([m, size], dtype=A.dtype, block_shape=A.block_shape, context=A.context)
    # distributed matrix which contains the first size rows of VT (the right singular vectors)
    VT = core.DistributedMatrix([size, n], dtype=A.dtype, block_shape=A.block_shape, context=A.context)
    # array of size size. Contains the singular values of A sorted in descending order
    s = np.empty(size, dtype=util.real_equiv(A.dtype))

    jobu = 'V' if compute_u else 'N'
    jobvt = 'V' if compute_v else 'N'
    args = [jobu, jobvt, m, n, A, s, U, VT]

    call_table = {'S': (ll.psgesvd, args + [ll.WorkArray('S')]),
                  'D': (ll.pdgesvd, args + [ll.WorkArray('D')]),
                  'C': (ll.pcgesvd, args + [ll.WorkArray('C')] + [np.zeros(1 + 4*sizeb, dtype=np.float32)]),
                  'Z': (ll.pzgesvd, args + [ll.WorkArray('Z')] + [np.zeros(1 + 4*sizeb, dtype=np.float64)])}

    func, args = call_table[A.sc_dtype]
    info = func(*args)

    if info < 0:
        raise core.ScalapackException("Failure.")

    if compute_u:
        if compute_v:
            return U, s, VT
        else:
            return U, s
    else:
        if compute_v:
            return s, VT
        else:
            return s


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


def triinv(A, lower=False, unit_triangular=False, overwrite_a=True):
    """Computes the inverse of a triangular distributed matrix.

    Computes the inverse of a real or complex upper or lower triangular
    distributed matrix.

    Parameters
    ----------
    A : DistributedMatrix
        The matrix to inverse.
    lower : boolean, optional
        True if `A` is lower triangular, else upper triangular (the default).
        The other triangular part of `A` is not referenced.
    unit_triangular : boolean, optional
        True if `A` is unit triangular (with 1 on the diagonal), else
        non-unit triangular (the default).
    overwrite_a : boolean, optional
        By default the input matrix is destroyed, if set to False a
        copy is taken and operated on.

    Returns
    -------
    inv : DistributedMatrix
        The inverse of `A`.
    """

    # Check if matrix is square
    util.assert_square(A)

    A = A if overwrite_a else A.copy()

    N, N = A.global_shape

    uplo = 'L' if lower else 'U'
    diag = 'U' if unit_triangular else 'N'

    args = [uplo, diag, N, A]

    call_table = {'S': (ll.pstrtri, args),
                  'D': (ll.pdtrtri, args),
                  'C': (ll.pctrtri, args),
                  'Z': (ll.pztrtri, args)}

    func, args = call_table[A.sc_dtype]
    info = func(*args)

    if info < 0:
        raise core.ScalapackException("Failure.")
    elif info > 0:
        raise core.ScalapackException("The triangular matrix is singular and its inverse can not be computed.")

    return A


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


def pinv2(A, overwrite_a=True, cond=None, rcond=None, return_rank=False, check_finite=True):
    """
    Compute the (Moore-Penrose) pseudo-inverse of a distributed matrix.

    Calculate a generalized inverse of a distributed matrix using its
    singular-value decomposition and including all 'large' singular
    values.

    This need not require that `A` must have full rank.

    Parameters
    ----------
    A : DistributedMatrix
        Matrix to be pseudo-inverted.
    overwrite_a : boolean, optional
        By default the input matrix is destroyed, if set to False a
        copy is taken and operated on.
    cond, rcond : float or None
        Cutoff for 'small' singular values.
        Singular values smaller than ``rcond*largest_singular_value``
        are considered zero.
        If None or -1, suitable machine precision is used.
    return_rank : bool, optional
        if True, return the effective rank of the matrix
    check_finite : boolean, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    B : DistributedMatrix
        The pseudo-inverse of matrix `A`.
    rank : int
        The effective rank of the matrix.  Returned if return_rank == True

    """

    # if check_finite:
    #     a = np.asarray_chkfinite(a)
    # else:
    #     a = np.asarray(a)
    A = A if overwrite_a else A.copy()

    U, s, VH = svd(A, overwrite_a=overwrite_a)

    if rcond is not None:
        cond = rcond
    if cond in [None,-1]:
        t = s.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}
        cond = factor[t] * np.finfo(t).eps

    rank = np.sum(s > cond * np.max(s))
    psigma_diag = 1.0 / s[: rank]

    B = dot(U[:, :rank] * psigma_diag, VH[:rank]).H

    if return_rank:
        return B, rank
    else:
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
