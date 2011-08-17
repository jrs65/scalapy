=============
 PyScalapack
=============

PyScalapack, as the name implies, is a wrapping of Scalapack such that it can be called by Python in a friendly manner.

Operations are performed on DistributedMatrix and DistributedVector objects which can be easily created whilst hiding all the nasty details of block cyclic distribution.

Currently this package is limited to double precision. Removing this restriction is on the cards.


Dependencies
============

This package depends upon two python packages ``numpy`` and ``mpi4py``. It is written largely in ``Cython`` and so requires that. Obviously it also requires an MPI distribution (OpenMPI and IntelMPI supported out the box), and a ScaLapack installation (Intel MKL currently supported, change setup.py to support others).

Installation
============

To build just go into pyscalapack/ and run the ::
> python setup.py build_ext --inplace

and then point your python path at the root directory. You may need to modify setup.py to change the MPI or Scalapack version.

Building this is a little bit of an art, as Cython forces the use of the compiler that built python itself (when linking). This mean mpicc cannot simply be used, and we need to manually fetch the flags required for building with MPI. This is likely to be the source of any difficulties building.



 
