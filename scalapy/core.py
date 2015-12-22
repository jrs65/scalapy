"""
===========================================
Core (:mod:`scalapy.core`)
===========================================

.. currentmodule:: scalapy.core

This module contains the core of `scalapy`: a set of routines and classes to
describe the distribution of MPI processes involved in the computation, and
interface with ``BLACS``; and a class which holds a block cyclic distributed
matrix for computation.


Routines
========

.. autosummary::
    :toctree: generated/

    initmpi


Classes
=======

.. autosummary::
    :toctree: generated/

    ProcessContext
    DistributedMatrix
    ScalapyException
    ScalapackException

"""

from numbers import Number
import numpy as np

from mpi4py import MPI

import blockcyclic
import blacs
import mpi3util


class ScalapyException(Exception):
    """Error in scalapy."""
    pass


class ScalapackException(Exception):
    """Error in calling Scalapack."""
    pass


_context = None
_block_shape = None


# Map numpy type into MPI type
typemap = { np.float32: MPI.FLOAT,
            np.float64: MPI.DOUBLE,
            np.complex64: MPI.COMPLEX,
            np.complex128: MPI.COMPLEX16 }


def _chk_2d_size(shape, positive=True):
    # Check that the shape describes a valid 2D grid. Zero shape not allowed when positive = True.

    if len(shape) != 2:
        return False

    if positive:
        if shape[0] <= 0 or shape[1] <= 0:
            return False
    else:
        if shape[0] < 0 or shape[1] < 0:
            return False

    return True


def initmpi(gridshape=None, block_shape=[32, 32]):
    r"""Initialise Scalapack on the current process.

    This routine sets up the BLACS grid, and sets the default context
    for this process.

    Parameters
    ----------
    gridsize : array_like
        A two element list (or other tuple etc), containing the
        requested shape for the process grid e.g. `[nprow, npcol]`.
    block_shape : array_like, optional
        The default blocksize for new arrays. A two element, [`brow,
        bcol]` list.
    """

    global _context, _block_shape

    # Setup the default context
    _context = ProcessContext(gridshape)

    # Set default blocksize
    _block_shape = tuple(block_shape)




class ProcessContext(object):
    r"""Stores information about an MPI/BLACS process.

    Parameters
    ----------
    gridshape : array_like
        A two element list (or other tuple etc), containing the
        requested shape for the process grid e.g. [nprow, npcol].

    comm : mpi4py.MPI.Comm, optional
        The MPI communicator to create a BLACS context for. If comm=None,
        then use MPI.COMM_WORLD instead.

    Attributes
    ----------
    grid_shape
    grid_position
    mpi_comm
    blacs_context
    all_grid_positions
    all_mpi_ranks
    """

    _grid_shape = (1, 1)

    @property
    def grid_shape(self):
        """Process grid shape."""
        return self._grid_shape


    _grid_position = (0, 0)

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


    @property
    def all_grid_positions(self):
        """Returns shape (mpi_comm_size,2) array, such that (arr[i,0], arr[i,1]) gives the grid position of mpi task i."""
        return self._all_grid_positions


    @property
    def all_mpi_ranks(self):
        """Inverse of all_grid_positions: returns 2D array such that arr[i,j] gives the mpi rank at grid position (i,j)."""
        return self._all_mpi_ranks


    def __init__(self, grid_shape, comm=None):
        """Construct a BLACS context for the current process.
        """

        # MPI setup
        if comm is None:
            comm = MPI.COMM_WORLD

        self._mpi_comm = comm

        # Grid shape setup
        if not _chk_2d_size(grid_shape):
            raise ScalapyException("Grid shape invalid.")

        gs = grid_shape[0]*grid_shape[1]
        if gs != self.mpi_comm.size:
            raise ScalapyException("Gridshape must be equal to the MPI size.")

        self._grid_shape = tuple(grid_shape)

        # Initialise BLACS context
        ctxt = blacs.sys2blacs_handle(self.mpi_comm)
        self._blacs_context = blacs.gridinit(ctxt, self.grid_shape[0], self.grid_shape[1])

        blacs_info = blacs.gridinfo(self.blacs_context)
        blacs_size, blacs_pos = blacs_info[:2], blacs_info[2:]

        # Check we got the gridsize we wanted
        if blacs_size[0] != self.grid_shape[0] or blacs_size[1] != self.grid_shape[1]:
            raise ScalapyException("BLACS did not give requested gridsize (requested %s, got %s)."
                                   % (repr(self.grid_shape), repr(blacs_size)))

        # Set the grid position.
        self._grid_position = blacs_pos

        #
        # As far as I know, BLACS doesn't guarantee any specific association between MPI tasks and grid positions, so
        # we compute all_grid_positions using MPI_Allgather().
        #
        # (Alternate approach: move the call to MPI_Allgather to the all_grid_positions property, and cache the result.
        # This would have the advantage that MPI_Allgather() only gets called if needed, but the disadvantage that it
        # would hang if the first call to all_grid_positions() is from a serial context.)
        #
        t = np.array(self.grid_position)
        assert t.shape == (2,)
        self._all_grid_positions = np.zeros((self.mpi_comm.size,2), dtype=t.dtype)
        self.mpi_comm.Allgather(t, self._all_grid_positions)

        # Compute all_mpi_ranks from all_grid_positions
        self._all_mpi_ranks = np.zeros(self.grid_shape, dtype=np.int)
        self._all_mpi_ranks[self._all_grid_positions[:,0],self._all_grid_positions[:,1]] = np.arange(self.mpi_comm.size, dtype=np.int)


class DistributedMatrix(object):
    r"""A matrix distributed over multiple MPI processes.

    Parameters
    ----------
    global_shape : list of integers
        The size of the global matrix eg. ``[Nr, Nc]``.
    dtype : np.dtype, optional
        The datatype of the array. See `Notes`_ for the supported types.
    block_shape: list of integers, optional
        The blocking size, packed as ``[Br, Bc]``. If ``None`` uses the default blocking
        (set via :func:`initmpi`).
    context : ProcessContext, optional
        The process context. If not set uses the default (recommended).

    Attributes
    ----------
    local_array
    desc
    context
    dtype
    mpi_dtype
    sc_dtype
    global_shape
    local_shape
    block_shape

    Methods
    -------
    empty_like
    indices
    from_global_array
    to_global_array
    from_file
    to_file
    redistribute


    .. _notes:

    Notes
    -----
    The type of the array must be specified with the standard numpy types. A
    :class:`DistributedMatrix` has properties for fetching the equivalent
    ``MPI`` (with :attr:`mpi_dtype`) and ``Scalapack`` types (which is a
    character given by :attr:`sc_dtype`).

    =================  =================  ==============  ===============================
    Numpy type         MPI type           Scalapack type  Description
    =================  =================  ==============  ===============================
    ``np.float32``     ``MPI.FLOAT``      ``S``           Single precision float
    ``np.float64``     ``MPI.DOUBLE``     ``D``           Double precision float
    ``np.complex64``   ``MPI.COMPLEX``    ``C``           Single precision complex number
    ``np.complex128``  ``MPI.COMPLEX16``  ``Z``           Double precision complex number
    =================  =================  ==============  ===============================

    Slicing
    -------

    Basic slicing is implemented for :class:`DistributedMatrix` objects allowing
    us to cut out sections of the global matrix, giving a new
    :class:`DistributedMatrix` containing the section and distributed over the
    same :class:`ProcessContext`. Note that this creates a *copy* of the
    original data, *not* a view of it. To use this simply do::

        dm = DistributedMatrix((10, 10), dtype=np.complex128)

        # Copy the first five columns
        five_cols = dm[:, :5]

        # Copy out the third and second to last rows
        two_rows = dm[-3:-1]

        # Copy every other row
        alternating_rows = dm[::2]

    """

    @property
    def local_array(self):
        """The local, block-cyclic packed segment of the matrix.

        This is an ndarray and is readonly. However, only the
        reference is readonly, the array itself can be modified in
        place.
        """
        if self._loccal_empty:
            return np.zeros(self.local_shape, order='F', dtype=self.dtype)
        else:
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
    def sc_dtype(self):
        """The Scalapack type as a character."""
        _sc_type = {np.float32: 'S',
                    np.float64: 'D',
                    np.complex64: 'C',
                    np.complex128: 'Z'}

        return _sc_type[self.dtype]


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

        """

        ## Check and set data type
        if dtype not in typemap.keys():
            raise Exception("Requested dtype not supported by Scalapack.")

        self._dtype = dtype

        ## Check and set global_shape
        if not _chk_2d_size(global_shape, positive=False):
            raise ScalapyException("Array global shape invalid.")

        self._global_shape = tuple(global_shape)

        ## Check and set default block_shape
        if not _block_shape and not block_shape:
            raise ScalapyException("No supplied or default blocksize.")

        block_shape = block_shape if block_shape else _block_shape

        # Validate block_shape.
        if not _chk_2d_size(block_shape):
            raise ScalapyException("Block shape invalid.")

        self._block_shape = block_shape

        ## Check and set context.
        if not context and not _context:
            raise ScalapyException("No supplied or default context.")
        self._context = context if context else _context

        # Allocate the local array.
        self._loccal_empty = True if self.local_shape[0] == 0 or self.local_shape[1] == 0 else False
        if self._loccal_empty:
            # as f2py can not handle zero sized array, we have to create an non-empty local array
            self._local_array = np.zeros(1, dtype=dtype)
        else:
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
        if self.global_shape[0] == 0 or self.global_shape[1] == 0:
            self._darr_f = None
            self._darr_c = None
            self._darr_list = []
        else:
            rank = self.context.mpi_comm.rank
            size = self.context.mpi_comm.size

            # Create distributed array view (F-ordered)
            self._darr_f = self.mpi_dtype.Create_darray(size, rank,
                                self.global_shape,
                                [MPI.DISTRIBUTE_CYCLIC, MPI.DISTRIBUTE_CYCLIC],
                                self.block_shape, self.context.grid_shape,
                                MPI.ORDER_F)
            self._darr_f.Commit()

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


    @classmethod
    def empty_trans(cls, mat):
        r"""Create a DistributedMatrix, with the same blocking
        but transposed shape as `mat`.

        Parameters
        ----------
        mat : DistributedMatrix
            The matrix to operate.

        Returns
        -------
        tmat : DistributedMatrix
        """
        return cls([mat.global_shape[1], mat.global_shape[0]], block_shape=mat.block_shape,
                   dtype=mat.dtype, context=mat.context)


    @classmethod
    def identity(cls, n, dtype=np.float64, block_shape=None, context=None):
        """Returns distributed n-by-n distributed matrix.

        Parameters
        ----------
        n : integer
           matrix size
        dtype : np.dtype, optional
           The datatype of the array.
           See DistributedMatrix.__init__ docstring for supported types.
        block_shape: list of integers, optional
           The blocking size, packed as ``[Br, Bc]``. If ``None`` uses the default blocking
           (set via :func:`initmpi`).
        context : ProcessContext, optional
           The process context. If not set uses the default (recommended).
        """

        ret = cls(global_shape = (n,n),
                  dtype = dtype,
                  block_shape = block_shape,
                  context = context)

        (g,r,c) = ret.local_diagonal_indices()

        ret.local_array[r,c] = 1.0
        return ret


    def copy(self):
        """Create a copy of this DistributedMatrix.

        This includes a full copy of the local data. However, the
        :attr:`context` is a reference to the original :class:`ProcessContext`.

        Returns
        -------
        copy : DistributedMatrix
        """
        cp = DistributedMatrix.empty_like(self)
        cp.local_array[:] = self.local_array

        return cp


    def row_indices(self):
        """The row indices of the global array local to the process.
        """
        return blockcyclic.indices_rc(self.global_shape[0],
                                      self.block_shape[0],
                                      self.context.grid_position[0],
                                      self.context.grid_shape[0])


    def col_indices(self):
        """The column indices of the global array local to the process.
        """
        return blockcyclic.indices_rc(self.global_shape[1],
                                      self.block_shape[1],
                                      self.context.grid_position[1],
                                      self.context.grid_shape[1])


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
        :math:`M_{ij} = i + j` can be created by::

            dm = DistributedMatrix(100, 100)
            rows, cols = dm.indices()
            dm.local_array[:] = rows + cols
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


    def local_diagonal_indices(self, allow_non_square=False):
        """Returns triple of 1D arrays (global_index, local_row_index, local_column_index).

        Each of these arrays has length equal to the number of elements on the global diagonal
        which are stored in the local matrix.  For each such element, global_index[i] is its
        position in the global diagonal, and (local_row_index[i], local_column_index[i]) gives
        its position in the local array.

        As an example of the use of these arrays, the global operation A_{ij} += i^2 delta_{ij}
        could be implemented with::

           (global_index, local_row_index, local_column_index) = A.local_diagonal_indices()
           A.local_array[local_row_index, local_column_index] += global_index**2
        """

        if (not allow_non_square) and (self.global_shape[0] != self.global_shape[1]):
            #
            # Attempting to access the "diagonal" of a non-square matrix probably indicates a bug.
            # Therefore we raise an exception unless the caller sets the allow_non_square flag.
            #
            raise RuntimeError('scalapy.core.DistributedMatrix.local_diagonal_indices() called on non-square matrix, and allow_non_square=False')

        ri, ci = map(blockcyclic.indices_rc,
                     self.global_shape,
                     self.block_shape,
                     self.context.grid_position,
                     self.context.grid_shape)

        global_index = np.intersect1d(ri, ci)

        (rank, local_row_index) = blockcyclic.localize_indices(global_index, self.block_shape[0], self.context.grid_shape[0])
        assert np.all(rank == self.context.grid_position[0])

        (rank, local_col_index) = blockcyclic.localize_indices(global_index, self.block_shape[1], self.context.grid_shape[1])
        assert np.all(rank == self.context.grid_position[1])

        return (global_index, local_row_index, local_col_index)


    def trace(self):
        """Returns global matrix trace (the trace is returned on all ranks)."""

        (g,r,c) = self.local_diagonal_indices()

        # Note: np.sum() returns 0 for length-zero array
        ret = np.array(np.sum(self.local_array[r,c]))
        self.context.mpi_comm.Allreduce(ret.copy(), ret, MPI.SUM)

        return ret

    @classmethod
    def from_global_array(cls, mat, rank=None, block_shape=None, context=None):

        r"""Create a DistributedMatrix directly from the global `array`.

        Parameters
        ----------
        mat : ndarray
            The global array to extract the local segments of.
        rank : integer
            Broadcast global matrix from given rank, to all ranks if set.
            Otherwise, if rank=None, assume all processes have a copy.
        block_shape: list of integers, optional
            The blocking size in [Br, Bc]. If `None` uses the default
            blocking (set via `initmpi`).
        context : ProcessContext, optional
            The process context. If not set uses the default (recommended).

        Returns
        -------
        dm : DistributedMatrix
        """
        # Broadcast if rank is not set.
        if rank is not None:
            comm = context.mpi_comm if context else _context.mpi_comm

            # Double check that rank is valid.
            if rank < 0 or rank >= comm.size:
                raise ScalapyException("Invalid rank.")

            if comm.rank == rank:
                if mat.ndim != 2:
                    raise ScalapyException("Array must be 2d.")

                mat = np.asfortranarray(mat)
                mat_shape = mat.shape
                mat_dtype = mat.dtype.type
            else:
                mat_shape = None
                mat_dtype = None

            mat_shape = comm.bcast(mat_shape, root=rank)
            mat_dtype = comm.bcast(mat_dtype, root=rank)

            m = cls(mat_shape, block_shape=block_shape, dtype=mat_dtype, context=context)

            if mat_shape[0] != 0 and mat_shape[1] != 0:
                # Each process should receive its local sections.
                rreq = comm.Irecv([m.local_array, m.mpi_dtype], source=rank, tag=0)

                if comm.rank == rank:
                    # Post each send
                    reqs = [ comm.Isend([mat, m._darr_list[dt]], dest=dt, tag=0)
                                 for dt in range(comm.size) ]

                    # Wait for requests to complete
                    MPI.Prequest.Waitall(reqs)

                rreq.Wait()

        else:
            if mat.ndim != 2:
                raise ScalapyException("Array must be 2d.")

            m = cls(mat.shape, block_shape=block_shape, dtype=mat.dtype.type, context=context)

            mat = np.asfortranarray(mat)
            m._load_array(mat)

        return m


    def _load_array(self, mat):
        ## Copy the local data out of the global mat.

        if (self.global_shape[0] == 0) or (self.global_shape[1] == 0):
            return

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

        if (self.global_shape[0] == 0) or (self.global_shape[1] == 0):
            return np.zeros(self.global_shape, dtype=self.dtype)

        comm = self.context.mpi_comm

        bcast = False
        if rank is None:
            rank = 0
            bcast = True

        # Double check that rank is valid.
        if rank < 0 or rank >= comm.size:
            raise ScalapyException("Invalid rank.")

        global_array = None
        if comm.rank == rank or bcast:
            global_array = np.zeros(self.global_shape, dtype=self.dtype, order='F')

        # Each process should send its local sections.
        sreq = comm.Isend([self.local_array, self.mpi_dtype], dest=rank, tag=0)

        if comm.rank == rank:
            # Post each receive
            reqs = [ comm.Irecv([global_array, self._darr_list[sr]], source=sr, tag=0)
                        for sr in range(comm.size) ]

            # Wait for requests to complete
            MPI.Prequest.Waitall(reqs)

        # Wait on send request. Important, as can get weird synchronisation
        # bugs otherwise as processes exit before completing their send.
        sreq.Wait()

        # Distribute to all processes if requested
        if bcast:
            comm.Bcast([global_array, self.mpi_dtype], root=rank)

        # Barrier to synchronise all processes
        comm.Barrier()

        return global_array


    def __iadd__(self, x):
        assert isinstance(x, DistributedMatrix)

        if self.global_shape != x.global_shape:
            raise RuntimeError("scalapy.DistributedMatrix.__iadd__: incompatible shapes")

        if ((self.block_shape != x.block_shape)
            or (self.context.grid_shape != x.context.grid_shape)
            or (self.context.grid_position != x.context.grid_position)):
            raise RuntimeError("scalapy.DistributedMatrix.__iadd__: for now, both matrices must have same blocking scheme")

        # Note: OK if dtypes don't match
        self.local_array[:] += x.local_array[:]

        return self


    def __mul__(self, x):
        if isinstance(x, DistributedMatrix):
            if self.global_shape != x.global_shape:
                raise RuntimeError("scalapy.DistributedMatrix.__mul__: incompatible shapes")

            if ((self.block_shape != x.block_shape)
                or (self.context.grid_shape != x.context.grid_shape)
                or (self.context.grid_position != x.context.grid_position)):
                raise RuntimeError("scalapy.DistributedMatrix.__mul__: for now, both matrices must have same blocking scheme")

            B = self.copy()

            # Note: OK if dtypes don't match
            B.local_array[:] *= x.local_array[:]

            return B

        elif isinstance(x, Number):
            B = self.copy()
            B.local_array[:] *= x

            return B

        elif isinstance(x, np.ndarray):
            if x.ndim != 1 and x.size != self.global_shape[1]:
                raise RuntimeError("scalapy.DistributedMatrix.__mul__: incompatible shapes")

            B = self.copy()
            for (i, col) in enumerate(self.col_indices()):
                B.local_array[:, i] *= x[col]

            return B
        else:
            raise RuntimeError('Unsupported type %s' % type(x))

    def _section(self, srow=0, nrow=None, scol=0, ncol=None):
        ## return a section [srow:srow+nrow, scol:scol+ncol] of the global array as a new distributed array
        nrow = self.global_shape[0] - srow if nrow is None else nrow
        ncol = self.global_shape[1] - scol if ncol is None else ncol
        assert nrow > 0 and ncol > 0, 'Invalid number of rows/columns: %d/%d' % (nrow, ncol)

        B = DistributedMatrix([nrow, ncol], dtype=self.dtype, block_shape=self.block_shape, context=self.context)

        args = [nrow, ncol, self._local_array , srow+1, scol+1, self.desc, B._local_array, 1, 1, B.desc, self.context.blacs_context]

        from . import lowlevel as ll

        call_table = {'S': (ll.psgemr2d, args),
                      'D': (ll.pdgemr2d, args),
                      'C': (ll.pcgemr2d, args),
                      'Z': (ll.pzgemr2d, args)}

        func, args = call_table[self.sc_dtype]
        func(*args)

        return B

    def _sec2sec(self, B, srowb=0, scolb=0, srow=0, nrow=None, scol=0, ncol=None):
        # copy a section [srow:srow+nrow, scol:scol+ncol] of the global array to
        # another distributed array B starting at (srowb, scolb)

        # Copy to the end of the row/column if the numbers are not set.
        nrow = self.global_shape[0] - srow if nrow is None else nrow
        ncol = self.global_shape[1] - scol if ncol is None else ncol

        # Check the number of rows and columns
        if nrow <= 0 or ncol <= 0:
            raise ScalapyException('Invalid number of rows/columns: %d/%d' % (nrow, ncol))

        # Set up the argument list
        args = [nrow, ncol,
                self._local_array, srow+1, scol+1, self.desc,
                B._local_array, srowb+1, scolb+1, B.desc,
                self.context.blacs_context]

        from . import lowlevel as ll

        call_table = {'S': (ll.psgemr2d, args),
                      'D': (ll.pdgemr2d, args),
                      'C': (ll.pcgemr2d, args),
                      'Z': (ll.pzgemr2d, args)}

        func, args = call_table[self.sc_dtype]
        func(*args)

    def __getitem__(self, items):
        # numpy-like global slicing operation, but returns a distributed array.
        #
        # Supports basic numpy slicing with start and stop, and positive step

        def swap(a, b):
            return b, a

        def regularize_idx(idx, N, axis):
            # Regularize an index to check it is valid
            idx1 = idx if idx >= 0 else idx + N
            if idx1 < 0 or idx1 >= N:
                raise IndexError('Index %d is out of bounds for axis %d with size %d' % (idx, axis, N))

            return 1, [(idx1, 1)]

        def regularize_slice(slc, N):
            # Regularize a slice object
            #
            # Takes a slice object and the axis length, and returns the total
            # number of elements in the slice and an array of (start, length)
            # tuples describing the blocks making up the slice

            start, stop, step = slc.start, slc.stop, slc.step
            step = step if step is not None else 1

            # Check the step
            if step == 0:
                raise ValueError('slice step cannot be zero')
            if step > 0:
                start = start if start is not None else 0
                stop = stop if stop is not None else N
            else:
                start = start if start is not None else N-1
                if stop is None:
                    stop_is_none = True
                else:
                    stop_is_none = False
                stop = stop if stop is not None else 0
            start = start if start >= 0 else start + N
            stop = stop if stop >= 0 else stop + N
            start = max(0, start)
            start = min(N, start)
            stop = max(0, stop)
            stop = min(N, stop)

            # If the step is 1 things are simple...
            if step == 1:
                m = stop - start
                if m > 0:
                    return m, [(start, m)]
                else:
                    return 0, []

            # ... if it is greater than one divide up into a series of blocks
            else:
                m = 0
                lst = []
                if step > 0:
                    while(start < stop):
                        lst.append((start, 1))
                        m += 1
                        start += step
                if step < 0:
                    while(start > stop):
                        lst.append((start, 1))
                        m += 1
                        start += step
                    if stop_is_none and start == stop:
                        m += 1
                        lst.append((0, 1))

                return m, lst


        nrow, ncol = self.global_shape

        # First replace any Ellipsis with a full slice(None, None, None) object, this
        # is fine because the matrix is always 2D and it vastly simplifies the logic
        if items is Ellipsis:
            items = slice(None, None, None)
        if items is tuple:
            items = tuple([slice(None, None, None) if items is Ellipsis else item for item in items])

        # First case deal with just a single slice (either an int or slice object)
        if type(items) in [int, long]:
            m, rows = regularize_idx(items, nrow, 0)
            n = ncol  # number of columns
            cols = [(0, ncol)]
        elif type(items) is slice:
            if items == slice(None, None, None):
                return self.copy()

            m, rows = regularize_slice(items, nrow)
            n = ncol
            cols = [(0, ncol)]

        # Then deal with the case of a tuple (i.e. slicing both dimensions)
        elif type(items) is tuple:

            # Check we have two indexed dimensions
            if len(items) != 2:
                raise ValueError('Two many indices for 2D matrix: %s' % repr(items))

            # Check the types in the slicing are correct
            if not ((type(items[0]) in [int, long, slice] or items[0] is Ellipsis) and
                    (type(items[1]) in [int, long, slice] or items[1] is Ellipsis)):
                raise ValueError('Invalid indices %s' % items)

            # Process case of wanting a specific row
            if type(items[0]) in [int, long]:
                m, rows = regularize_idx(items[0], nrow, 0)

                if type(items[1]) in [int, long]:
                    n, cols = regularize_idx(items[1], ncol, 1)
                elif type(items[1]) is slice:
                    n, cols = regularize_slice(items[1], ncol)
                else:
                    raise ValueError('Invalid indices %s' % items)

            # Case of wanting a slice of rows
            elif type(items[0]) is slice:
                m, rows = regularize_slice(items[0], nrow)

                if type(items[1]) in [int, long]:
                    n, cols = regularize_idx(items[1], ncol, 1)
                elif type(items[1]) is slice:
                    if items[0] == slice(None, None, None) and items[1] == slice(None, None, None):
                        return self.copy()

                    n, cols = regularize_slice(items[1], ncol)
                else:
                    raise ValueError('Invalid indices %s' % items)

            else:
                raise ValueError('Invalid indices %s' % items)
        else:
            raise ValueError('Invalid indices %s' % items)

        # Create output DistributedMatrix
        B = DistributedMatrix([m, n], dtype=self.dtype, block_shape=self.block_shape, context=self.context)
        srowb = 0

        # Iterate over blocks to copy from self to new output matrix
        for (srow, nrow) in rows:
            scolb = 0
            for (scol, ncol) in cols:
                self._sec2sec(B, srowb, scolb, srow, nrow, scol, ncol)
                scolb += ncol
            srowb += nrow

        return B


    def _copy_from_np(self, a, asrow=0, anrow=None, ascol=0, ancol=None, srow=0, scol=0, block_shape=None, rank=0):
        ## copy a section of a numpy array a[asrow:asrow+anrow, ascol:ascol+ancol] to self[srow:srow+anrow, scol:scol+ancol], once per block_shape

        Nrow, Ncol = self.global_shape
        srow = srow if srow >= 0 else srow + Nrow
        srow = max(0, srow)
        srow = min(srow, Nrow)
        scol = scol if scol >= 0 else scol + Ncol
        scol = max(0, scol)
        scol = min(scol, Ncol)
        if self.context.mpi_comm.rank == rank:
            if not (a.ndim == 1 or a.ndim == 2):
                raise ScalapyException('Unsupported high dimensional array.')

            a = np.asfortranarray(a.astype(self.dtype)) # type conversion
            a = a.reshape(-1, a.shape[-1]) # reshape to two dimensional
            am, an = a.shape
            asrow = asrow if asrow >= 0 else asrow + am
            asrow = max(0, asrow)
            asrow = min(asrow, am)
            ascol = ascol if ascol >= 0 else ascol + an
            ascol = max(0, ascol)
            ascol = min(ascol, an)
            m = am - asrow if anrow is None else anrow
            m = max(0, m)
            m = min(m, am - asrow, Nrow - srow)
            n = an - ascol if ancol is None else ancol
            n = max(0, n)
            n = min(n, an - ascol, Ncol - scol)
        else:
            m, n = 1, 1

        asrow = self.context.mpi_comm.bcast(asrow, root=rank)
        ascol = self.context.mpi_comm.bcast(ascol, root=rank)
        m = self.context.mpi_comm.bcast(m, root=rank) # number of rows to copy
        n = self.context.mpi_comm.bcast(n, root=rank) # number of columes to copy

        if m == 0 or n == 0:
            return self

        block_shape = self.block_shape if block_shape is None else block_shape
        if not _chk_2d_size(block_shape):
            raise ScalapyException("Invalid block_shape")

        bm, bn = block_shape
        br = blockcyclic.num_blocks(m, bm) # number of blocks for row
        bc = blockcyclic.num_blocks(n, bn) # number of blocks for column
        rm = m - (br - 1) * bm # remained number of rows of the last block
        rn = n - (bc - 1) * bn # remained number of columes of the last block

        # due to bugs in scalapy, it is needed to first init an process context here
        ProcessContext([1, self.context.mpi_comm.size], comm=self.context.mpi_comm) # process context

        for bri in range(br):
            M = bm if bri != br - 1 else rm
            for bci in range(bc):
                N = bn if bci != bc - 1 else rn
                if self.context.mpi_comm.rank == rank:
                    pc = ProcessContext([1, 1], comm=MPI.COMM_SELF) # process context
                    desc = self.desc
                    desc[1] = pc.blacs_context
                    desc[2], desc[3] = a.shape
                    desc[4], desc[5] = a.shape
                    desc[8] = a.shape[0]
                    args = [M, N, a, asrow+1+bm*bri, ascol+1+bn*bci, desc, self._local_array, srow+1+bm*bri, scol+1+bn*bci, self.desc, self.context.blacs_context]
                else:
                    desc = np.zeros(9, dtype=np.int32)
                    desc[1] = -1
                    args = [M, N, np.zeros(1, dtype=self.dtype) , asrow+1+bm*bri, ascol+1+bn*bci, desc, self._local_array, srow+1+bm*bri, scol+1+bn*bci, self.desc, self.context.blacs_context]

                from . import lowlevel as ll

                call_table = {'S': (ll.psgemr2d, args),
                              'D': (ll.pdgemr2d, args),
                              'C': (ll.pcgemr2d, args),
                              'Z': (ll.pzgemr2d, args)}

                func, args = call_table[self.sc_dtype]
                func(*args)

        return self


    def np2self(self, a, srow=0, scol=0, block_shape=None, rank=0):
        """Copy a one or two dimensional numpy array `a` owned by
        rank `rank` to the section of the distributed matrix starting
        at row `srow` and column `scol`. Once copy a section equal
        or less than `block_shape` if `a` is large.
        """
        return self._copy_from_np(a, asrow=0, anrow=None, ascol=0, ancol=None,srow=srow, scol=scol, block_shape=block_shape, rank=rank)


    def self2np(self, srow=0, nrow=None, scol=0, ncol=None, block_shape=None, rank=0):
        """Copy a section of the distributed matrix
        self[srow:srow+nrow, scol:scol+ncol] to a two dimensional numpy
        array owned by rank `rank`. Once copy a section equal or less
        than `block_shape` if the copied section is large.
        """
        Nrow, Ncol = self.global_shape
        srow = srow if srow >= 0 else srow + Nrow
        srow = max(0, srow)
        srow = min(srow, Nrow)
        scol = scol if scol >= 0 else scol + Ncol
        scol = max(0, scol)
        scol = min(scol, Ncol)
        m = Nrow - srow if nrow is None else nrow
        m = max(0, m)
        m = min(m, Nrow - srow)
        n = Ncol - scol if ncol is None else ncol
        n = max(0, n)
        n = min(n, Ncol - scol)

        if self.context.mpi_comm.rank == rank:
            a = np.empty((m, n), dtype=self.dtype, order='F')
        else:
            a = None

        if m == 0 or n == 0:
            return a

        block_shape = self.block_shape if block_shape is None else block_shape
        if not _chk_2d_size(block_shape):
            raise ScalapyException("Invalid block_shape")

        bm, bn = block_shape
        br = blockcyclic.num_blocks(m, bm) # number of blocks for row
        bc = blockcyclic.num_blocks(n, bn) # number of blocks for column
        rm = m - (br - 1) * bm # remained number of rows of the last block
        rn = n - (bc - 1) * bn # remained number of columes of the last block

        # due to bugs in scalapy, it is needed to first init an process context here
        ProcessContext([1, self.context.mpi_comm.size], comm=self.context.mpi_comm) # process context

        for bri in range(br):
            M = bm if bri != br - 1 else rm
            for bci in range(bc):
                N = bn if bci != bc - 1 else rn
                if self.context.mpi_comm.rank == rank:
                    pc = ProcessContext([1, 1], comm=MPI.COMM_SELF) # process context
                    desc = self.desc
                    desc[1] = pc.blacs_context
                    desc[2], desc[3] = a.shape
                    desc[4], desc[5] = a.shape
                    desc[8] = a.shape[0]
                    args = [M, N, self._local_array, srow+1+bm*bri, scol+1+bn*bci, self.desc, a , 1+bm*bri, 1+bn*bci, desc, self.context.blacs_context]
                else:
                    desc = np.zeros(9, dtype=np.int32)
                    desc[1] = -1
                    args = [M, N, self._local_array, srow+1+bm*bri, scol+1+bn*bci, self.desc, np.zeros(1, dtype=self.dtype) , 1+bm*bri, 1+bn*bci, desc, self.context.blacs_context]

                from . import lowlevel as ll

                call_table = {'S': (ll.psgemr2d, args),
                              'D': (ll.pdgemr2d, args),
                              'C': (ll.pcgemr2d, args),
                              'Z': (ll.pzgemr2d, args)}

                func, args = call_table[self.sc_dtype]
                func(*args)

        return a


    @classmethod
    def from_file(cls, filename, global_shape, dtype, block_shape=None, context=None, order='F', displacement=0):
        """Read in a global array from a file.

        Parameters
        ----------
        filename : string
            Name of file to read in.
        global_shape : [nrows, ncols]
            Shape of global array.
        dtype : numpy datatype
            Datatype of array.
        block_shape : [brows, bcols]
            Shape of block, if None, try to use default size.
        context : ProcessContext
            Description of process distribution.
        order : 'F' or 'C'
            Storage order on disk.
        displacement : integer
            Displacement from the start of file (in bytes)

        Returns
        -------
        dm : DistributedMatrix
        """

        if (global_shape[0] == 0) or (global_shape[1] == 0):
            return None

        # Initialise DistributedMatrix
        dm = cls(global_shape, dtype=dtype, block_shape=block_shape, context=context)

        # Open the file, and read out the segments
        f = MPI.File.Open(dm.context.mpi_comm, filename, MPI.MODE_RDONLY)
        f.Set_view(displacement, dm.mpi_dtype, dm._darr_f, "native")
        f.Read_all(dm.local_array)
        f.Close()

        return dm


    def to_file(self, filename, order='F', displacement=0):
        """Write a DistributedMatrix out to a file.

        Parameters
        ----------
        filename : string
            Name of file to write to.
        """

        if (self.global_shape[0] == 0) or (self.global_shape[1] == 0):
            return

        # Open the file, and read out the segments
        f = MPI.File.Open(self.context.mpi_comm, filename, MPI.MODE_RDWR | MPI.MODE_CREATE)

        filelength = displacement + mpi3util.type_get_extent(self._darr_f)[1]  # Extent is index 1

        # Preallocate to ensure file is long enough for writing.
        f.Preallocate(filelength)

        # Set view and write out.
        f.Set_view(displacement, self.mpi_dtype, self._darr_f, "native")
        f.Write_all(self.local_array)
        f.Close()

    def redistribute(self, block_shape=None, context=None):
        """Redistribute a matrix with another grid or block shape.

        Parameters
        ----------
        block_shape : [brows, bcols], optional
            New block shape. If `None` use the current block shape.
        context : ProcessContext, optional
            New process context. Must be over the same MPI communicator. If
            `None` use the current communicator.

        Returns
        -------
        dm : DistributedMatrix
            Newly distributed matrix.
        """

        # Check that we are actually redistributing across something
        if (block_shape is None) and (context is None):
            import warnings
            warnings.warn("Neither block_shape or context is set.")

        # Fix up default parameters
        if block_shape is None:
            block_shape = self.block_shape
        if context is None:
            context = self.context

        # Check that we are redistributing over the same communicator
        if context.mpi_comm != self.context.mpi_comm:
            raise ScalapyException("Can only redsitribute over the same MPI communicator.")

        from . import lowlevel as ll

        dm = DistributedMatrix(self.global_shape, dtype=self.dtype, block_shape=block_shape, context=context)

        args = [self.global_shape[0], self.global_shape[1], self, dm, self.context.blacs_context]

        # Prepare call table
        call_table = {'S': (ll.psgemr2d, args),
                      'D': (ll.pdgemr2d, args),
                      'C': (ll.pcgemr2d, args),
                      'Z': (ll.pzgemr2d, args)}

        # Call routine
        func, args = call_table[self.sc_dtype]
        func(*args)

        return dm


    def transpose(self):
        """Transpose the distributed matrix."""

        trans = DistributedMatrix.empty_trans(self)

        args = [self.global_shape[1], self.global_shape[0], 1.0, self, 0.0, trans]

        from . import lowlevel as ll

        call_table = {'S': (ll.pstran, args),
                      'D': (ll.pdtran, args),
                      'C': (ll.pctranu, args),
                      'Z': (ll.pztranu, args)}

        func, args = call_table[self.sc_dtype]
        func(*args)

        return trans


    @property
    def T(self):
        """Transpose the distributed matrix."""
        return self.transpose()


    def conj(self):
        """Complex conjugate the distributed matrix."""

        # if real
        if self.sc_dtype in ['S', 'D']:
            return self

        # if complex
        cj = DistributedMatrix.empty_like(self)
        cj.local_array[:] = self.local_array.conj()

        return cj


    @property
    def C(self):
        """Complex conjugate the distributed matrix."""
        return self.conj()


    def hconj(self):
        """Hermitian conjugate the distributed matrix, i.e., transpose
        and complex conjugate the distributed matrix."""

        # if real
        if self.sc_dtype in ['S', 'D']:
            return self.transpose()

        # if complex
        hermi = DistributedMatrix.empty_trans(self)

        args = [self.global_shape[1], self.global_shape[0], 1.0, self, 0.0, hermi]

        from . import lowlevel as ll

        call_table = {'C': (ll.pctranc, args),
                      'Z': (ll.pztranc, args)}

        func, args = call_table[self.sc_dtype]
        func(*args)

        return hermi


    @property
    def H(self):
        """Hermitian conjugate the distributed matrix, i.e., transpose
        and complex conjugate the distributed matrix."""
        return self.hconj()
