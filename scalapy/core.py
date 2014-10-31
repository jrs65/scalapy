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

    if shape[0] <= 0 or shape[1] <= 0:
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
        

    @staticmethod
    def _local_indices(global_indices, block_size, grid_size):
        """Converts an array of global row/col indices to an array of local indices.

        This helper method is used in DistributedMatrix.local_diagonal_indices() and DistributedMatrix.transpose().

        Parameters
        ----------
        global_indices : np.ndarray
            Integer-valued array of global row/col indices
        block_size : integer
            Usually DistributedMatrix.block_shape[0] or DistributedMatrix.block_shape[1]
        grid_size : integer
            Usually ProcessContext.grid_shape[0] or ProcessContext.grid_shape[1]

        Returns
        -------
        local_indices : np.ndarray
        """

        return np.mod(global_indices, block_size) + block_size * np.divide(global_indices, block_size*grid_size)


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
        local_row_index = self._local_indices(global_index, self.block_shape[0], self.context.grid_shape[0])
        local_col_index = self._local_indices(global_index, self.block_shape[1], self.context.grid_shape[1])

        return (global_index, local_row_index, local_col_index)


    def trace(self):
        """Returns global matrix trace (the trace is returned on all cores)."""

        (g,r,c) = self.local_diagonal_indices()
        
        # Note: np.sum() returns 0 for length-zero array
        ret = np.array(np.sum(self.local_array[r,c]))
        self.context.mpi_comm.Allreduce(ret.copy(), ret, MPI.SUM)        

        return ret
        

    def transpose(self, block_shape=None, context=None):
        """Returns distributed matrix transpose."""

        if block_shape is None:
            block_shape = self.block_shape
        if context is None:
            context = self.context

        if self.context.mpi_comm is not context.mpi_comm:
            raise RuntimeError('input/output contexts in DistributedMatrix.transpose() must have the same MPI communicator')

        (m,n) = self.global_shape

        src_rindices = [ blockcyclic.indices_rc(m, self.block_shape[0], p, self.context.grid_shape[0]) for p in xrange(self.context.grid_shape[0]) ]
        src_cindices = [ blockcyclic.indices_rc(n, self.block_shape[1], p, self.context.grid_shape[1]) for p in xrange(self.context.grid_shape[1]) ]
        dst_rindices = [ blockcyclic.indices_rc(n, block_shape[0], p, context.grid_shape[0]) for p in xrange(context.grid_shape[0]) ]
        dst_cindices = [ blockcyclic.indices_rc(m, block_shape[1], p, context.grid_shape[1]) for p in xrange(context.grid_shape[1]) ]

        send_rindices = [ ]
        for ci in dst_cindices:
            t = np.intersect1d(ci, src_rindices[self.context.grid_position[0]])
            t = self._local_indices(t, self.block_shape[0], self.context.grid_shape[0])
            send_rindices.append(t)

        send_cindices = [ ]
        for ri in dst_rindices:
            t = np.intersect1d(ri, src_cindices[self.context.grid_position[1]])
            t = self._local_indices(t, self.block_shape[1], self.context.grid_shape[1])
            send_cindices.append(t)

        send_counts = np.array([ len(send_rindices[q]) * len(send_cindices[p]) for (p,q) in context.all_grid_positions ])
        send_displs = np.concatenate(([0], np.cumsum(send_counts[:-1])))
        send_buf = np.zeros(np.sum(send_counts), dtype=self.dtype)

        for q in xrange(context.grid_shape[1]):
            a = self.local_array[send_rindices[q],:]
            for p in xrange(context.grid_shape[0]):
                b = a[:,send_cindices[p]]
                si = send_displs[context.all_mpi_ranks[p,q]]
                sn = send_counts[context.all_mpi_ranks[p,q]]
                send_buf[si:si+sn] = np.reshape(np.transpose(b), (-1,))    # note transpose here

        del a,b    # save memory by dropping references

        recv_rindices = [ ]
        for ci in src_cindices:
            t = np.intersect1d(ci, dst_rindices[context.grid_position[0]])
            t = self._local_indices(t, block_shape[0], context.grid_shape[0])
            recv_rindices.append(t)

        recv_cindices = [ ]
        for ri in src_rindices:
            t = np.intersect1d(ri, dst_cindices[context.grid_position[1]])
            t = self._local_indices(t, block_shape[1], context.grid_shape[1])
            recv_cindices.append(t)

        recv_counts = np.array([ len(recv_rindices[q]) * len(recv_cindices[p]) for (p,q) in self.context.all_grid_positions ])
        recv_displs = np.concatenate(([0], np.cumsum(recv_counts[:-1])))
        recv_buf = np.zeros(np.sum(recv_counts), dtype=self.dtype)

        self.context.mpi_comm.Alltoallv((send_buf, (send_counts, send_displs)),
                                        (recv_buf, (recv_counts, recv_displs)))
        
        del send_buf   # save memory

        ret = DistributedMatrix(global_shape=(n,m),
                                dtype = self.dtype,
                                block_shape = block_shape,
                                context = context)        

        for q in xrange(self.context.grid_shape[1]):
            a = np.zeros((len(recv_rindices[q]),ret.local_array.shape[1]), dtype=self.dtype)
            for p in xrange(self.context.grid_shape[0]):
                ri = recv_displs[self.context.all_mpi_ranks[p,q]]
                rn = recv_counts[self.context.all_mpi_ranks[p,q]]
                a[:,recv_cindices[p]] = np.reshape(recv_buf[ri:ri+rn], (len(recv_rindices[q]),len(recv_cindices[p])))
            ret.local_array[recv_rindices[q],:] = a

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

            mat = comm.bcast(mat, root=rank)
        else:
            if mat.ndim != 2:
                raise ScalapyException("Array must be 2d.")

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
