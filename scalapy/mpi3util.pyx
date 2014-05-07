"""Useful routines from MPI-3 that are not available in `mpi4py`.
"""

from mpi4py import MPI
from mpi4py cimport MPI

include "mpi4py/mpi.pxi"

## MPI-3 Routines routines
cdef extern from "mpi.h":

    ctypedef size_t MPI_Count

    int MPI_Type_size_x(MPI_Datatype datatype, MPI_Count *size)
    int MPI_Type_get_extent_x(MPI_Datatype datatype, MPI_Count *lb, MPI_Count *extent)


def type_size(datatype):
    """Fetch size of the MPI Datatype.

    This directly wraps `MPI_Type_size_x` from MPI-3 which does not have
    32-bit issues unlike `MPI_Type_size`.

    Parameters
    ----------
    datatype : MPI.Datatype
        MPI derived datatype.

    Returns
    -------
    size : integer
        Size of type in bytes.
    """
    cdef size_t size

    err = MPI_Type_size_x(<MPI_Datatype>(<MPI.Datatype>datatype).ob_mpi, &size)

    if err == 0:
        retval = size
        return retval
    else:
        raise Exception("Could not fetch type size.")


def type_get_extent(datatype):
    """Fetch extent of the MPI Datatype.

    This directly wraps `MPI_Type_get_extent_x` from MPI-3 which does not have
    32-bit issues unlike `MPI_Type_get_extent`.

    Parameters
    ----------
    datatype : MPI.Datatype
        MPI derived datatype.

    Returns
    -------
    lb : integer
        Lower bound of type in bytes.
    extent : integer
        Extent of type in bytes.
    """
    cdef size_t lb, extent

    err = MPI_Type_get_extent_x(<MPI_Datatype>(<MPI.Datatype>datatype).ob_mpi, &lb, &extent)

    if err == 0:
        retval = (lb, extent)
        return retval
    else:
        raise Exception("Could not fetch type extent.")
