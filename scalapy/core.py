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
