
from mpi4py import MPI

from mpi4py cimport MPI
from blacs cimport *


class BLACSException(Exception):
    pass


def sys2blacs_handle(comm):
    """Create a BLACS handle from an MPI Communicator.

    Parameters
    ----------
    comm : MPI.Comm

    Returns
    -------
    ctxt : integer
        BLACS context handle.
    """
    if not isinstance(comm, MPI.Comm):
        raise Exception("Did not recieve MPI Communicator.")

    return Csys2blacs_handle(<MPI_Comm>(<MPI.Comm>comm).ob_mpi)


def gridinfo(ctxt):
    """Fetch the process grid info.

    Parameters
    ----------
    ctxt : integer
        BLACS context int.

    Returns
    -------
    nrows, ncols : integer
        Total size of grid.
    row, col : integer
        Position in grid.

    Raises
    ------
    BLACSException
        If process grid undefined.
    """
    cdef int ictxt, nrows, ncols, row, col

    ictxt = <int>ctxt

    Cblacs_gridinfo(ictxt, &nrows, &ncols, &row, &col)

    if nrows == -1:
        raise BLACSException("Grid not defined.")

    return (nrows, ncols, row, col)


def gridinit(ctxt, nrows, ncols, order="Row"):
    """Initialise the BLACS process grid.

    Parameters
    ----------
    ctxt : integer
        BLACS context handle. Must have been initialised with
        `sys2blacs_handle`.
    nrows, ncols : integer
        Process grid size.

    Raises
    ------
    BLACSException
        If grid initialisation failed.
    """
    cdef int ictxt

    ictxt = <int>ctxt

    # Ensure order is property converted to an ASCII string
    order = bytes(order.encode('ascii')) if isinstance(order, str) else order

    if order != b"Row":
        raise Exception("Order not valid.")

    # Initialise the grid
    Cblacs_gridinit(&ictxt, order, nrows, ncols)

    # Check that the initialisation worked
    try:
        gridinfo(ctxt)
    except BLACSException:
        raise BLACSException("Grid initialisation failed.")

    # Get new context
    ctxt = ictxt

    return ctxt
