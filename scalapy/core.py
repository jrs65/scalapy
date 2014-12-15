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


def _chk_2d_size(shape):
    # Check that the shape describes a valid 2D grid.

    if len(shape) != 2:
        return False

    if shape[0] < 0 or shape[1] < 0:
        return False

    return True


def initmpi(gridshape, block_shape=None, comm=None):
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
    comm : mpi4py.MPI.Comm, optional
        The MPI communicator to create a BLACS context for. If comm=None,
        then use MPI.COMM_WORLD instead.
    """

    global _context, _block_shape

    # Setup the default context
    _context = ProcessContext(gridshape, comm=comm)

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
        if not _chk_2d_size(global_shape):
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


    def local_diagonal_indices(self, allow_non_square=False):
        """Returns triple of 1D arrays (global_index, local_row_index, local_column_index).

        Each of these arrays has length equal to the number of elements on the global diagonal
        which are stored in the local matrix.  For each such element, global_index[i] is its
        position in the global diagonal, and (local_row_index[i], local_column_index[i]) gives
        its position in the local array.

        As an example of the use of these arrays, the global operation A_{ij} += i^2 delta_{ij}
        could be implemented with:

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
        """Returns global matrix trace (the trace is returned on all cores)."""

        (g,r,c) = self.local_diagonal_indices()

        # Note: np.sum() returns 0 for length-zero array
        ret = np.array(np.sum(self.local_array[r,c]))
        self.context.mpi_comm.Allreduce(ret.copy(), ret, MPI.SUM)

        return ret


    # def transpose(self, block_shape=None, context=None):
    #     """Returns distributed matrix transpose."""

    #     if block_shape is None:
    #         block_shape = self.block_shape
    #     if context is None:
    #         context = self.context

    #     if self.context.mpi_comm is not context.mpi_comm:
    #         raise RuntimeError('input/output contexts in DistributedMatrix.transpose() must have the same MPI communicator')

    #     (m,n) = self.global_shape

    #     src_rindices = [ blockcyclic.indices_rc(m, self.block_shape[0], p, self.context.grid_shape[0]) for p in xrange(self.context.grid_shape[0]) ]
    #     src_cindices = [ blockcyclic.indices_rc(n, self.block_shape[1], p, self.context.grid_shape[1]) for p in xrange(self.context.grid_shape[1]) ]
    #     dst_rindices = [ blockcyclic.indices_rc(n, block_shape[0], p, context.grid_shape[0]) for p in xrange(context.grid_shape[0]) ]
    #     dst_cindices = [ blockcyclic.indices_rc(m, block_shape[1], p, context.grid_shape[1]) for p in xrange(context.grid_shape[1]) ]

    #     send_rindices = [ ]
    #     for ci in dst_cindices:
    #         t = np.intersect1d(ci, src_rindices[self.context.grid_position[0]])
    #         t = blockcyclic.localize_indices(t, self.block_shape[0], self.context.grid_shape[0])[1]
    #         send_rindices.append(t)

    #     send_cindices = [ ]
    #     for ri in dst_rindices:
    #         t = np.intersect1d(ri, src_cindices[self.context.grid_position[1]])
    #         t = blockcyclic.localize_indices(t, self.block_shape[1], self.context.grid_shape[1])[1]
    #         send_cindices.append(t)

    #     send_counts = np.array([ len(send_rindices[q]) * len(send_cindices[p]) for (p,q) in context.all_grid_positions ])
    #     send_displs = np.concatenate(([0], np.cumsum(send_counts[:-1])))
    #     send_buf = np.zeros(np.sum(send_counts), dtype=self.dtype)

    #     for q in xrange(context.grid_shape[1]):
    #         a = self.local_array[send_rindices[q],:]
    #         for p in xrange(context.grid_shape[0]):
    #             b = a[:,send_cindices[p]]
    #             si = send_displs[context.all_mpi_ranks[p,q]]
    #             sn = send_counts[context.all_mpi_ranks[p,q]]
    #             send_buf[si:si+sn] = np.reshape(np.transpose(b), (-1,))    # note transpose here

    #     del a,b    # save memory by dropping references

    #     recv_rindices = [ ]
    #     for ci in src_cindices:
    #         t = np.intersect1d(ci, dst_rindices[context.grid_position[0]])
    #         t = blockcyclic.localize_indices(t, block_shape[0], context.grid_shape[0])[1]
    #         recv_rindices.append(t)

    #     recv_cindices = [ ]
    #     for ri in src_rindices:
    #         t = np.intersect1d(ri, dst_cindices[context.grid_position[1]])
    #         t = blockcyclic.localize_indices(t, block_shape[1], context.grid_shape[1])[1]
    #         recv_cindices.append(t)

    #     recv_counts = np.array([ len(recv_rindices[q]) * len(recv_cindices[p]) for (p,q) in self.context.all_grid_positions ])
    #     recv_displs = np.concatenate(([0], np.cumsum(recv_counts[:-1])))
    #     recv_buf = np.zeros(np.sum(recv_counts), dtype=self.dtype)

    #     self.context.mpi_comm.Alltoallv((send_buf, (send_counts, send_displs)),
    #                                     (recv_buf, (recv_counts, recv_displs)))

    #     del send_buf   # save memory

    #     ret = DistributedMatrix(global_shape=(n,m),
    #                             dtype = self.dtype,
    #                             block_shape = block_shape,
    #                             context = context)

    #     for q in xrange(self.context.grid_shape[1]):
    #         a = np.zeros((len(recv_rindices[q]),ret.local_array.shape[1]), dtype=self.dtype)
    #         for p in xrange(self.context.grid_shape[0]):
    #             ri = recv_displs[self.context.all_mpi_ranks[p,q]]
    #             rn = recv_counts[self.context.all_mpi_ranks[p,q]]
    #             a[:,recv_cindices[p]] = np.reshape(recv_buf[ri:ri+rn], (len(recv_rindices[q]),len(recv_cindices[p])))
    #         ret.local_array[recv_rindices[q],:] = a

    #     return ret


    def get_rows(self, rows):
        r"""Return selected rows of a DistributedMatrix, as a new Distributed Matrix (i.e. moral equivalent of self[rows,:]).

        FIXME wouldn't it be nice to define a __getitem__ which would allow general row/column slicing?

        Parameters
        ----------
           rows : 1D numpy array (must be the same on all tasks)
        """

        (m,n) = self.global_shape
        B = self.block_shape[0]               # row block length
        P = self.context.grid_shape[0]        # number of processes in row grid
        (p0,q0) = self.context.grid_position  # this task's position in grid
        nc = self.local_array.shape[1]        # number of local columns

        rows = np.array(rows)
        assert rows.ndim==1
        assert np.issubdtype(rows.dtype, np.integer)
        assert np.all(rows >= 0) and np.all(rows < m)

        # gri[p][i] = global row index corresponding to (rank, output_local_index) = (p,i)
        k = len(rows)   # output matrix will be k-by-n
        gri = [ blockcyclic.indices_rc(k, B, p, P) for p in xrange(P) ]
        gri = [ rows[g] for g in gri ]

        # (rrk,lri) = (input_rank, input_local_index) pair corresponding to (rank, output_local_index) = (p,i)
        rrk = [ ]
        lri = [ ]
        for g in gri:
            (r,l) = blockcyclic.localize_indices(g, B, P)
            rrk.append(r)
            lri.append(l)

        # send_indices[p] = input_local_indices to be sent to rank p
        send_indices = [ l[np.nonzero(r==p0)] for (r,l) in zip(rrk,lri) ]

        # recv_indices[p] = output_local_indices to be received from row rank p
        recv_indices = [ np.nonzero(rrk[p0]==p)[0] for p in xrange(P) ]

        # per-row block counts and displacements, in units of "rows"
        scounts = np.array([ len(x) for x in send_indices ])
        rcounts = np.array([ len(x) for x in recv_indices ])
        sdispls = np.concatenate(([0], np.cumsum(scounts[:-1])))
        rdispls = np.concatenate(([0], np.cumsum(rcounts[:-1])))

        # per-mpi-task counts and displacements, in units of "matrix elements"
        mpi_scounts = np.array([ (scounts[p]*nc if q==q0 else 0)  for (p,q) in self.context.all_grid_positions ])
        mpi_rcounts = np.array([ (rcounts[p]*nc if q==q0 else 0)  for (p,q) in self.context.all_grid_positions ])
        mpi_sdispls = np.array([ (sdispls[p]*nc if q==q0 else 0)  for (p,q) in self.context.all_grid_positions ])
        mpi_rdispls = np.array([ (rdispls[p]*nc if q==q0 else 0)  for (p,q) in self.context.all_grid_positions ])

        sbuf = np.zeros(np.sum(scounts)*nc, dtype=self.dtype)

        # pack sbuf
        for (d,c,si) in zip(sdispls, scounts, send_indices):
            sbuf[d*nc:(d+c)*nc] = np.reshape(self.local_array[si,:], (-1,))

        rbuf = np.zeros(np.sum(rcounts)*nc, dtype=self.dtype)

        self.context.mpi_comm.Alltoallv((sbuf, (mpi_scounts, mpi_sdispls)),
                                        (rbuf, (mpi_rcounts, mpi_rdispls)))

        del sbuf

        ret = DistributedMatrix((k,n), dtype=self.dtype, block_shape=self.block_shape, context=self.context)
        assert ret.local_array.shape[1] == nc

        # unpack rbuf
        for (d,c,ri) in zip(rdispls, rcounts, recv_indices):
            ret.local_array[ri,:] = np.reshape(rbuf[d*nc:(d+c)*nc], (c,nc))

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

        args = [nrow, ncol, self.local_array , srow+1, scol+1, self.desc, B.local_array, 1, 1, B.desc, self.context.blacs_context]

        from . import lowlevel as ll

        call_table = {'S': (ll.psgemr2d, args),
                      'D': (ll.pdgemr2d, args),
                      'C': (ll.pcgemr2d, args),
                      'Z': (ll.pzgemr2d, args)}

        func, args = call_table[self.sc_dtype]
        ll.expand_args = False
        func(*args)
        ll.expand_args = True

        return B


    def _sec2sec(self, B, srowb=0, scolb=0, srow=0, nrow=None, scol=0, ncol=None):
        ## copy a section [srow:srow+nrow, scol:scol+ncol] of the global array to another distributed array B starting at (srowb, scolb)
        nrow = self.global_shape[0] - srow if nrow is None else nrow
        ncol = self.global_shape[1] - scol if ncol is None else ncol
        assert nrow > 0 and ncol > 0, 'Invalid number of rows/columns: %d/%d' % (nrow, ncol)

        args = [nrow, ncol, self.local_array , srow+1, scol+1, self.desc, B.local_array, srowb+1, scolb+1, B.desc, self.context.blacs_context]

        from . import lowlevel as ll

        call_table = {'S': (ll.psgemr2d, args),
                      'D': (ll.pdgemr2d, args),
                      'C': (ll.pcgemr2d, args),
                      'Z': (ll.pzgemr2d, args)}

        func, args = call_table[self.sc_dtype]
        ll.expand_args = False
        func(*args)
        ll.expand_args = True


    def __getitem__(self, items):
        ## numpy array like sling operation, but return a distributed array

        def swap(a, b):
            return b, a

        if type(items) in [int, long]:
            assert items >= 0, 'Negative index %d' % items
            assert items < self.global_shape[0], 'Invalid index %d' % items
            srow = items # start row
            scol = 0     # start column
            m = 1        # number of rows
            n = self.global_shape[1] # number of columns
            rows = [(srow, m)]
            cols = [(scol, n)]
        elif type(items) is slice:
            start, stop, step = items.start, items.stop, items.step
            start = start if start is not None else 0
            stop = stop if stop is not None else self.global_shape[0]
            step = step if step is not None else 1
            assert abs(step) > 0, 'Invalid step 0'
            if step < 0:
                step = -step
                start, stop = swap(stat, stop)

            if step == 1:
                assert start < stop, 'Invalid indices %s' % items
                m = stop - start
                n = self.global_shape[1]
                rows = [(start, m)]
                cols = [(0, n)]
            else:
                raise Exception('Not implemented yet')
        elif items is Ellipsis:
            return self.copy()
        elif type(items) is tuple:
            assert len(items) == 2, 'Invalid indices %s' % items
            assert type(items[0]) in [int, long, slice] or items[0] is Ellipsis and type(items[1]) in [int, long, slice] or items[1] is Ellipsis, 'Invalid indices %s' % items
            if type(items[0]) is slice:
                start1, stop1, step1 = items[0].start, items[0].stop, items[0].step
                start1 = start1 if start1 is not None else 0
                stop1 = stop1 if stop1 is not None else self.global_shape[0]
                step1 = step1 if step1 is not None else 1
                assert abs(step1) > 0, 'Invalid step 0'
                if step1 < 0:
                    step1 = -step1
                    start1, stop1 = swap(start1, stop1)

                if step1 == 1:
                    assert start1 < stop1, 'Invalid indices %s' % items[0]
                    m = stop1 - start1
                    rows = [(start1, m)]
                else:
                    raise Exception('Not implemented yet')

                if type(items[1]) is slice:
                    start2, stop2, step2 = items[1].start, items[1].stop, items[1].step
                    start2 = start2 if start2 is not None else 0
                    stop2 = stop2 if stop2 is not None else self.global_shape[1]
                    step2 = step2 if step2 is not None else 1
                    assert abs(step2) > 0, 'Invalid step 0'
                    if step2 < 0:
                        step2 = -step2
                        start2, stop2 = swap(start2, stop2)

                    if step2 == 1:
                        assert start2 < stop2, 'Invalid indices %s' % items[1]
                        n = stop2 - start2
                        cols = [(start2, n)]
                    else:
                        raise Exception('Not implemented yet')

                else:
                    raise Exception('Not implemented yet')

            else:
                raise Exception('Not implemented yet')
        else:
            raise Exception('Invalid indices %s' % items)

        B = DistributedMatrix([m, n], dtype=self.dtype, block_shape=self.block_shape, context=self.context)
        srowb = 0
        scolb = 0
        for (srow, nrow) in rows:
            for (scol, ncol) in cols:
                self._sec2sec(B, srowb, scolb, srow, nrow, scol, ncol)
                scolb += ncol
            srowb += nrow

        return B


    def _copy_from_np(self, a, asrow=0, anrow=None, ascol=0, ancol=None, srow=0, scol=0, block_shape=None, rank=0):
        ## copy a section of a numpy array a[asrow:asrow+anrow, ascol:ascol+ancol] to self[srow:srow+anrow, scol:scol+ancol], once per block_shape
        if self.context.mpi_comm.rank == rank:
            assert a.ndim == 1 or a.ndim == 2, 'Unsupported high dimensional array.'
            a = np.asfortranarray(a.astype(self.dtype)) # type conversion
            a = a.reshape(-1, a.shape[-1]) # reshape to two dimensional
            am, an = a.shape
            assert 0 <= asrow < am, 'Invalid start row index asrow: %s' % asrow
            assert 0 <= ascol < an, 'Invalid start column index ascol: %s' % ascol
            m = am - asrow if anrow is None else anrow
            n = an - ascol if ancol is None else ancol
            assert 0 < m <= am - asrow, 'Invalid number of rows anrow: %s' % anrow
            assert 0 < n <= an - ascol, 'Invalid number of columes ancol: %s' % ancol
        else:
            m, n = 1, 1

        m = self.context.mpi_comm.bcast(m, root=rank) # number of rows to copy
        n = self.context.mpi_comm.bcast(n, root=rank) # number of columes to copy

        assert 0 <= srow < self.global_shape[0], 'Invalid start row index srow: %s' % srow
        assert 0 <= scol < self.global_shape[1], 'Invalid start column index scol: %s' % scol
        assert 0 < m <= self.global_shape[0] - srow, 'Invalid number of rows anrow: %s' % anrow
        assert 0 < n <= self.global_shape[1] - scol, 'Invalid number of columes ancol: %s' % ancol

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
                    args = [M, N, a, asrow+1+bm*bri, ascol+1+bn*bci, desc, self.local_array, srow+1+bm*bri, scol+1+bn*bci, self.desc, self.context.blacs_context]
                else:
                    desc = np.zeros(9, dtype=np.int32)
                    desc[1] = -1
                    args = [M, N, np.zeros(1, dtype=self.dtype) , asrow+1+bm*bri, ascol+1+bn*bci, desc, self.local_array, srow+1+bm*bri, scol+1+bn*bci, self.desc, self.context.blacs_context]

                from . import lowlevel as ll

                call_table = {'S': (ll.psgemr2d, args),
                              'D': (ll.pdgemr2d, args),
                              'C': (ll.pcgemr2d, args),
                              'Z': (ll.pzgemr2d, args)}

                func, args = call_table[self.sc_dtype]
                ll.expand_args = False
                func(*args)
                ll.expand_args = True

        return self


    def np2self(self, a, srow=0, scol=0, block_shape=None, rank=0):
        """Copy a one or two dimensional numpy array `a` owned by
        rank `rank` to the section of the distributed matrix starting
        at row `srow` and column `scol`. Once copy a section equal
        or less than `block_shape` if `a` is large.
        """
        return self._copy_from_np(a, asrow=0, anrow=None, ascol=0, ancol=None,srow=srow, scol=scol, block_shape=block_shape, rank=rank)


    def self2np(self, srow=0, nrow=None, scol=0, ncol=None, rank=0):
        """Copy a section of the distributed matrix
        self[srow:srow+nrow, scol:scol+ncol] to a two dimensional numpy
        array owned by rank `rank`.
        """
        assert 0 <= srow < self.global_shape[0], 'Invalid start row index srow: %s' % srow
        assert 0 <= scol < self.global_shape[1], 'Invalid start column index scol: %s' % scol
        m = self.global_shape[0] - srow if nrow is None else nrow
        n = self.global_shape[1] - scol if ncol is None else ncol
        assert 0 < m <= self.global_shape[0] - srow, 'Invalid number of rows anrow: %s' % nrow
        assert 0 < n <= self.global_shape[1] - scol, 'Invalid number of columes ancol: %s' % ncol

        # due to bugs in scalapy, it is needed to first init an process context here
        ProcessContext([1, self.context.mpi_comm.size], comm=self.context.mpi_comm) # process context

        if self.context.mpi_comm.rank == rank:
            a = np.empty((m, n), dtype=self.dtype, order='F')
            pc = ProcessContext([1, 1], comm=MPI.COMM_SELF) # process context
            desc = self.desc
            desc[1] = pc.blacs_context
            desc[2], desc[3] = a.shape
            desc[4], desc[5] = a.shape
            desc[8] = a.shape[0]
            args = [m, n, self.local_array, srow+1, scol+1, self.desc, a , 1, 1, desc, self.context.blacs_context]
        else:
            a = None
            desc = np.zeros(9, dtype=np.int32)
            desc[1] = -1
            args = [m, n, self.local_array, srow+1, scol+1, self.desc, np.zeros(1, dtype=self.dtype) , 1, 1, desc, self.context.blacs_context]

        from . import lowlevel as ll

        call_table = {'S': (ll.psgemr2d, args),
                      'D': (ll.pdgemr2d, args),
                      'C': (ll.pcgemr2d, args),
                      'Z': (ll.pzgemr2d, args)}

        func, args = call_table[self.sc_dtype]
        ll.expand_args = False
        func(*args)
        ll.expand_args = True

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

        if (self.global_shape[0] == 0) or (self.global_shape[1] == 0):
            return

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
