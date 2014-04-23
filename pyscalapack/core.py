import numpy as np

from mpi4py import MPI

import blockcyclic
import blacs


class PyScalapackException(Exception):
    pass

_context = None
_block_shape = None


# Map numpy type into MPI type
typemap = { np.float32 : MPI.FLOAT,
            np.float64 : MPI.DOUBLE,
            np.complex64: MPI.COMPLEX,
            np.complex128 : MPI.COMPLEX16 }


def _chk_2d_size(shape):
    # Check that the shape describes a valid 2D grid.

    if len(shape) != 2:
        return False

    if shape[0] <= 0 or shape[1] <= 0:
        return False

    return True




def initmpi(gridshape, block_shape=None):
    r"""Initialise Scalapack on the current process.

    This routine sets up the BLACS grid, and sets the default context
    for this process.

    Parameters
    ----------
    gridsize : array_like
        A two element list (or other tuple etc), containing the
        requested shape for the process grid e.g. `[nprow, npcol]`.
    blocksize : array_like, optional
        The default blocksize for new arrays. A two element, [`brow,
        bcol]` list.
    """

    global _context, _block_shape

    # Setup the default context
    _context = ProcessContext(gridshape)

    # Set default blocksize
    _block_shape = block_shape




class ProcessContext(object):
    r"""Stores information about an MPI/BLACS process.
    """

    _grid_shape = [1, 1]

    @property
    def grid_shape(self):
        """Process grid shape."""
        return self._grid_shape


    _grid_position = [0, 0]

    @property
    def grid_position(self):
        """Process grid position."""
        return self._grid_position


    _mpi_comm = None

    @property
    def mpi_comm(self):
        """MPI Communicator for this ProcessContext."""
        return self._mpi_comm


    _blacs_context = None

    @property
    def blacs_context(self):
        """BLACS context handle."""
        return self._blacs_context


    def __init__(self, grid_shape, comm=None):
        """Construct a BLACS context for the current process.

        Creates a BLACS context for the given MPI Communicator.

        Parameters
        ----------
        gridshape : array_like
            A two element list (or other tuple etc), containing the
            requested shape for the process grid e.g. [nprow, npcol].

        comm : mpi4py.MPI.Comm, optional
            The MPI communicator to create a BLACS context for. If comm=None,
            then use MPI.COMM_WORLD instead.
        """

        # MPI setup
        if comm is None:
            comm = MPI.COMM_WORLD

        self._mpi_comm = comm

        # Grid shape setup
        if not _chk_2d_size(grid_shape):
            raise PyScalapackException("Grid shape invalid.")

        gs = grid_shape[0]*grid_shape[1]
        if gs != self.mpi_comm.size:
            raise PyScalapackException("Gridshape must be equal to the MPI size.")

        self._grid_shape = grid_shape

        # Initialise BLACS context
        self._blacs_context = blacs.sys2blacs_handle(self.mpi_comm)

        blacs.gridinit(self.blacs_context, self.grid_shape[0], self.grid_shape[1])

        blacs_info = blacs.gridinfo(self.blacs_context)
        blacs_size, blacs_pos = blacs_info[:2], blacs_info[2:]

        # Check we got the gridsize we wanted
        if blacs_size[0] != self.grid_shape[0] or blacs_size[1] != self.grid_shape[1]:
            raise PyScalapackException("BLACS did not give requested gridsize.")

        # Set the grid position.
        self._grid_position = blacs_pos



class DistributedMatrix(object):
    r"""A matrix distributed over multiple MPI processes.
    """

    @property
    def local_array(self):
        """The local, block-cyclic packed segment of the matrix.

        This is an ndarray and is readonly. However, only the
        reference is readonly, the array itself can be modified in
        place.
        """
        return self._local_array


    @property
    def desc(self):
        """The Scalapack array descriptor. See [1]_. Returned as an integer
        ndarray and is readonly.

        .. [1] http://www.netlib.org/scalapack/slug/node77.html
        """
        return self._desc.copy()


    @property
    def context(self):
        """The ProcessContext of this matrix."""
        return self._context


    @property
    def dtype(self):
        """The numpy datatype of this matrix."""
        return self._dtype


    @property
    def mpi_dtype(self):
        """The base MPI Datatype."""
        return typemap[self.dtype]


    @property
    def global_shape(self):
        """The shape of the global matrix."""
        return self._global_shape


    @property
    def local_shape(self):
        """The shape of the local matrix."""

        lshape = map(blockcyclic.numrc, self.global_shape,
                     self.block_shape, self.context.grid_position,
                     self.context.grid_shape)

        return tuple(lshape)


    @property
    def block_shape(self):
        """The blocksize for the matrix."""
        return self._block_shape


    def __init__(self, global_shape, dtype=np.float64, block_shape=None, context=None):
        r"""Initialise an empty DistributedMatrix.

        Parameters
        ----------
        global_shape : list of integers
            The size of the global matrix eg. [Nr, Nc].
        dtype : np.dtype, optional
            The datatype of the array. Only `float32`, `float64` (default) and
            `complex128` are supported by Scalapack.
        block_shape: list of integers, optional
            The blocking size, packed as [Br, Bc]. If `None` uses the default blocking
            (set via `initmpi`).
        context : ProcessContext, optional
            The process context. If not set uses the default (recommended). 
        """

        ## Check and set data type
        if dtype not in typemap.keys():
            raise Exception("Requested dtype not supported by Scalapack.")

        self._dtype = dtype

        ## Check and set global_shape
        if not _chk_2d_size(global_shape):
            raise PyScalapackException("Array global shape invalid.")

        self._global_shape = global_shape

        ## Check and set default block_shape
        if not _block_shape and not block_shape:
            raise PyScalapackException("No supplied or default blocksize.")

        block_shape = block_shape if block_shape else _block_shape

        # Validate block_shape.
        if not _chk_2d_size(block_shape):
            raise PyScalapackException("Block shape invalid.")

        self._block_shape = block_shape

        ## Check and set context.
        if not context and not _context:
            raise PyScalapackException("No supplied or default context.")
        self._context = context if context else _context

        # Allocate the local array.
        self._local_array = np.zeros(self.local_shape, order='F', dtype=dtype)

        # Create the descriptor
        self._mkdesc()

        # Create the MPI distributed datatypes
        self._mk_mpi_dtype()


    def _mkdesc(self):
        # Make the Scalapack array descriptor
        self._desc = np.zeros(9, dtype=np.int32)

        self._desc[0] = 1  # Dense matrix
        self._desc[1] = self.context.blacs_context
        self._desc[2] = self.global_shape[0]
        self._desc[3] = self.global_shape[1]
        self._desc[4] = self.block_shape[0]
        self._desc[5] = self.block_shape[1]
        self._desc[6] = 0
        self._desc[7] = 0
        self._desc[8] = self.local_shape[0]


    def _mk_mpi_dtype(self):
        ## Construct the MPI datatypes (both Fortran and C ordered)
        ##   These are required for reading in and out of arrays and files.

        # Get MPI process info
        rank = self.context.mpi_comm.rank
        size = self.context.mpi_comm.size

        # Create distributed array view (F-ordered)
        self._darr_f = self.mpi_dtype.Create_darray(size, rank, 
                            self.global_shape,
                            [MPI.DISTRIBUTE_CYCLIC, MPI.DISTRIBUTE_CYCLIC],
                            self.block_shape, self.context.grid_shape,
                            MPI.ORDER_F).Commit()

        # Create distributed array view (F-ordered)
        self._darr_c = self.mpi_dtype.Create_darray(size, rank, 
                            self.global_shape,
                            [MPI.DISTRIBUTE_CYCLIC, MPI.DISTRIBUTE_CYCLIC],
                            self.block_shape, self.context.grid_shape,
                            MPI.ORDER_C).Commit()

        # Create list of types for all ranks (useful for passing to global array)
        self._darr_list = [ self.mpi_dtype.Create_darray(size, ri, 
                                self.global_shape,
                                [MPI.DISTRIBUTE_CYCLIC, MPI.DISTRIBUTE_CYCLIC],
                                self.block_shape, self.context.grid_shape,
                                MPI.ORDER_F).Commit() for ri in range(size) ]



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
        return cls(mat.global_shape, block_shape=mat.block_shape,
                   dtype=mat.dtype, context=mat.context)


    def indices(self, full=True):
        r"""The indices of the elements stored in the local matrix.

        This can be used to easily build up distributed matrices that
        depend on their co-ordinates.

        Parameters
        ----------
        full : boolean, optional
            If False the matrices of indices are not fleshed out, if True the
            full matrices are returned. This is like the difference between
            np.ogrid and np.mgrid.

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
        >>> dm.local_array[:] = rows + cols
        """

        ri, ci = map(blockcyclic.indices_rc,
                     self.global_shape,
                     self.block_shape,
                     self.context.grid_position,
                     self.context.grid_shape)

        ri = ri.reshape((-1, 1), order='F')
        ci = ci.reshape((1, -1), order='F')

        if full:
            ri, ci = np.broadcast_arrays(ri, ci)
            ri = np.asfortranarray(ri)
            ci = np.asfortranarray(ci)

        return (ri, ci)


    @classmethod
    def from_global_array(cls, mat, block_shape=None, context=None):

        r"""Create a DistributedMatrix directly from the global `array`.

        Parameters
        ----------
        mat : ndarray
            The global array to extract the local segments of.
        blocksize: list of integers, optional
            The blocking size in [Br, Bc]. If `None` uses the default
            blocking (set via `initmpi`).
        context : ProcessContext, optional
            The process context. If not set uses the default (recommended).

        Returns
        -------
        dm : DistributedMatrix
        """
        if mat.ndim != 2:
            raise PyScalapackException("Array must be 2d.")

        m = cls(mat.shape, block_shape=block_shape, dtype=mat.dtype.type, context=context)

        matf = np.asfortranarray(mat)
        m._load_array(matf)

        return m


    def _load_array(self, mat):
        ## Copy the local data out of the global mat.

        self._darr_f.Pack(mat, self.local_array[:], 0, self.context.mpi_comm)


    def to_global_array(self, rank=None):
        """Copy distributed data into a global array.

        This is mainly intended for testing. Would be a bad idea for larger problems.

        Parameters
        ----------
        rank : integer, optional
            If rank is None (default) then gather onto all nodes. If rank is
            set, then gather only onto one node.

        Returns
        -------
        matrix : np.ndarray
            The global matrix.
        """
        comm = self.context.mpi_comm

        bcast = False
        if rank is None:
            rank = 0
            bcast = True

        # Double check that rank is valid.
        if rank < 0 or rank >= comm.size:
            raise PyScalapackException("Invalid rank.")

        global_array = None
        if comm.rank == rank or bcast:
            global_array = np.zeros(self.global_shape, dtype=self.dtype, order='F')

        # Each process should send its local sections.
        comm.Isend([self.local_array, self.mpi_dtype], dest=rank, tag=0)

        if comm.rank == rank:
            # Post each receive
            reqs = [ comm.Irecv([global_array, self._darr_list[sr]], source=sr, tag=0)
                        for sr in range(comm.size) ]

            # Wait for requests to complete
            MPI.Prequest.Waitall(reqs)

        # Distribute to all processes if requested
        if bcast:
            comm.Bcast([global_array, self.mpi_dtype], root=rank)

        return global_array
