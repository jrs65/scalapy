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

import numpy as np

from . import core


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

    import h5py

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
