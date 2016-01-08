"""
==========================================
HDF5 Utilities (:mod:`~scalapy.hdf5utils`)
==========================================

Useful routines for dealing with HDF5 files.

Routines
========

.. autosummary::
    :toctree: generated/

    allocate_hdf5_dataset
"""
from __future__ import print_function, division, absolute_import

import h5py
import numpy as np

from . import core
from . import blockcyclic

def ensure_hdf5_dataset(fname, dsetname, shape, dtype, create=False):
    """Ensure a HDF5 dataset exists and return its offset and size.

    If it exists, the dataset must have been created contiguously and be
    allocated. If either the file, or the dataset does not exist (and `create`
    is set), it will be created, however the dataset not be filled.

    Parameters
    ----------
    fname : string
        Name of the file to write.
    dsetname : string
        Name of the dataset to write (must be at root level).
    shape : tuple
        Shape of the dataset.
    dtype : numpy datatype
        Type of the dataset.
    create : boolean, optional
        If the dataset does

    Returns
    -------
    offset : integer
        Offset into the file at which the dataset starts (in bytes).
    size : integer
        Size of the dataset in bytes.
    """

    # Create/open file
    with h5py.File(fname, 'a' if create else 'r') as f:
        ## Check if dset does not exist, and create if we need to
        if dsetname not in f:
            # If not, create it.
            if create:
                # Create dataspace and HDF5 datatype
                sp = h5py.h5s.create_simple(shape, shape)
                tp = h5py.h5t.py_create(dtype)

                # Create a new plist and tell it to allocate the space for dataset
                # immediately, but don't fill the file with zeros.
                plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
                plist.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
                plist.set_fill_time(h5py.h5d.FILL_TIME_NEVER)

                # Create the dataset
                dset = h5py.h5d.create(f.id, dsetname, tp, sp, plist)

                # Get the offset of the dataset into the file.
                state = dset.get_offset(), dset.get_storage_size()
            else:
                raise core.ScalapackException("Dataset does not exist.")

        ## If the dataset does exist, check that it is suitable, and return its info
        else:
            dataset = f[dsetname]
            dset = dataset.id
            # Check to ensure dataset is not chunked
            if dataset.chunks is not None:
                raise core.ScalapyException('Cannot access chunked dataset.')

            if dataset.shape != tuple(shape):
                raise core.ScalapyException('Dataset shape does not match.')

            # Check that dataset is long enough
            requested_size = np.prod(shape) * np.dtype(dtype).itemsize
            if dset.get_storage_size() < requested_size:
                raise core.ScalapyException('Allocated dataset size is too small.')

            # Fetch the dataset information
            dset = dataset.id
            state = dset.get_offset(), dset.get_storage_size()

    return state


def write_matrix(a, f, dataset_name, root=0, memlimit_gb=1.0, nblocks=None):
    """Low-tech routine which writes a matrix to an hdf5 dataset.

    This is an alternative to the "high-tech" approach (scalapy.hdf5utils.ensure_hdf5_dataset
    followed by scalapy.core.DistributedMatrix.to_file()).

    It gathers the matrix onto the root task (in blocks, to avoid exceeding memory limits)
    then writes the file from the root.

    Parameters
    ----------
    a : numpy.ndarray (serial case) or scalapy.core.DistributedMatrix (parallel)
        Matrix to write
    f : h5py.File (or h5py.Group) object
        File object (should be None on non-root tasks)
    dataset_name : string
        Name of hdf5 dataset which will be created by this routine
    root : integer
        Root MPI task
    memlimit_gb : float
        If nblocks is not specified, it will be chosen to fit within this memory cap
    nblocks : integer, or None
        Number of blocks to use when gathering matrix onto root task
        (if None, then memlimit_gb will determine nblocks)
    """

    if isinstance(a, np.ndarray) and len(a.shape)==2:
        f.create_dataset(dataset_name, data=a)
        return

    if not isinstance(a, core.DistributedMatrix):
        raise RuntimeError('write_matrix: expected either rank-2 numpy array or scalapy.core.DistributedMatrix')

    mpi_rank = a.context.mpi_comm.rank
    nrows_global = a.global_shape[0]
    ncols_global = a.global_shape[1]

    if nblocks is None:
        # Factor 2 here is because of double-buffering needed to unpack the result of mpi_gather (see below)
        gb_per_row = 2 * ncols_global * np.dtype(a.dtype).itemsize / 1.0e9
        rows_per_block = int(memlimit_gb / gb_per_row)

        if rows_per_block == 0:
            raise RuntimeError('write_matrix: memlimit_gb is too small')

        nblocks = (nrows_global + rows_per_block - 1) // rows_per_block

    rindices = [ blockcyclic.indices_rc(nrows_global, a.block_shape[0], p, a.context.grid_shape[0]) for p in xrange(a.context.grid_shape[0]) ]
    cindices = [ blockcyclic.indices_rc(ncols_global, a.block_shape[1], p, a.context.grid_shape[1]) for p in xrange(a.context.grid_shape[1]) ]

    col_counts = np.array([ len(c) for c in cindices ])
    col_displs = np.concatenate(([0], np.cumsum(col_counts[:-1])))

    assert nblocks > 0
    assert np.sum(col_counts) == ncols_global
    assert 0 <= root < a.context.mpi_comm.size

    if mpi_rank == root:
        assert isinstance(f, h5py.Group)
        dset = f.create_dataset(dataset_name, (nrows_global,ncols_global), dtype=a.dtype)

    for b in xrange(nblocks):
        # (gri:grj) = global row range processed in this block
        gri = (b * nrows_global) // nblocks
        grj = ((b+1) * nrows_global) // nblocks

        # (lri:lrj) = local row ranges processed in this block
        lri = [ np.searchsorted(r,gri,side='left') for r in rindices ]
        lrj = [ np.searchsorted(r,grj,side='left') for r in rindices ]
        my_lri = lri[a.context.grid_position[0]]
        my_lrj = lrj[a.context.grid_position[0]]

        # counts, displs, buffers for mpi_gather operation
        row_counts = [ (j-i) for (i,j) in zip(lri,lrj) ]
        row_displs = np.concatenate(([0], np.cumsum(row_counts[:-1])))
        mpi_counts = np.array([ (row_counts[p]*col_counts[q]) for (p,q) in a.context.all_grid_positions ])
        mpi_displs = np.array([ (row_displs[p]*ncols_global + row_counts[p]*col_displs[q]) for (p,q) in a.context.all_grid_positions ])
        rbuf = np.zeros((grj-gri)*ncols_global, dtype=a.dtype) if (mpi_rank==root) else None
        sbuf = np.reshape(a.local_array[my_lri:my_lrj,:], (-1,))   # note: zero-copy

        # mpi_gatherv fails if sbuf is noncontiguous (note: not actually sure why
        # this is necessary; I expected slices of the general form a.local_array[i:j,:]
        # to be contiguous)
        sbuf = np.ascontiguousarray(sbuf)

        # some end-to-end checks on the above calculations
        assert np.sum(row_counts) == (grj-gri)
        assert np.sum(mpi_counts) == (grj-gri) * ncols_global
        assert sbuf.size == mpi_counts[mpi_rank]

        for (r,i,j) in zip(rindices, lri, lrj):
            assert 0 <= i <= j <= len(r)
            assert (i == j) or (gri <= r[i] < grj)
            assert (i == j) or (gri <= r[j-1] < grj)
            assert (i == 0) or (r[i-1] < gri)
            assert (j == len(r)) or (r[j] >= grj)

        # now ready for mpi_gather!
        a.context.mpi_comm.Gatherv(sbuf, (rbuf,(mpi_counts,mpi_displs)), root=root)
        del sbuf

        if mpi_rank != root:
            continue

        #
        # Unpack rbuf into 2D array of shape (grj-gri, ncols_global)
        # Step 1: unpack rows, leaving columns in packed form
        #
        rbuf2 = np.zeros((grj-gri,ncols_global), dtype=a.dtype)
        for (rk,(p,q)) in enumerate(a.context.all_grid_positions):
            # source array is an m-by-n matrix at displacement d
            (m,n,d) = (row_counts[p], col_counts[q], mpi_displs[rk])

            # destination consists of row list rl, contiguous columns w/displacement cd
            rl = rindices[p][lri[p]:lrj[p]] - gri
            cd = col_displs[q]

            rbuf2[rl,cd:(cd+n)] = np.reshape(rbuf[d:(d+m*n)], (m,n))

        #
        # Unpacking step 2: permute cols
        #
        del rbuf
        rbuf3 = np.zeros((grj-gri,ncols_global), dtype=a.dtype)
        for (d,n,cl) in zip(col_displs, col_counts, cindices):
            rbuf3[:,cl] = rbuf2[:,d:(d+n)]

        # Write to file
        del rbuf2
        dset[gri:grj,:] = rbuf3
        del rbuf3
