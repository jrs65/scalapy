"""
================================================
scalapy Utilities (:mod:`~pyscalapack.util`)
================================================


Routines
========

.. autosummary::
    :toctree: generated/

    flatten
"""

import numpy as np

from . import core


def flatten(x):
    """Flatten a set of nested list and tuples.

    Returns a single, flat list which contains all elements retrieved
    from the sequence and all recursively contained sub-sequences
    (iterables).

    Parameters
    ----------
    x : list or tuple
        Set of lists and tuples to flatten.

    Returns
    -------
    flat : list or tuple

    Examples
    --------

    >>> [1, 2, [3,4], (5,6)]
    [1, 2, [3, 4], (5, 6)]
    >>> flatten([[[1,2,3], (42,None)], [4,5], [6], 7, MyVector(8,9,10)])
    [1, 2, 3, 42, None, 4, 5, 6, 7, 8, 9, 10]"""

    result = []
    for el in x:
        if isinstance(el, (list, tuple)):
        #if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def assert_square(A):
    """Assert that a distributed matrix is square.
    """
    Alist = flatten([A])
    for A in Alist:

        gs = A.global_shape
        if gs[0] != gs[1]:
            raise core.ScalapyException("Matrix must be square (has dimensions %i x %i)." % (gs[0], gs[1]))


def real_equiv(dtype):
    ## Return the real datatype with the same precision as dtype.
    if dtype == np.float32 or dtype == np.complex64:
        return np.float32

    if dtype == np.float64 or dtype == np.complex128:
        return np.float64

    raise core.ScalapyException("Unsupported data type.")
