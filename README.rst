========
 scalapy
========

``scalapy`` is a wrapping of Scalapack such that it can be called by Python in
a friendly manner.

Operations are performed on ``DistributedMatrix`` objects which can be easily
created whilst hiding all the nasty details of block cyclic distribution.


Dependencies
============

This package depends upon two python packages ``numpy`` and ``mpi4py``. It is
written largely in pure Python, but some parts require ``Cython`` and
``f2py``. Obviously it also requires an ``MPI`` distribution (OpenMPI and
IntelMPI supported out the box), and a ``Scalapack`` installation (both Intel
MKL and NETLIB are currently supported, change setup.py to select).

Installation
============

To build just go edit ``setup.py`` to choose the correct options, and then
run::

    $ python setup.py install

Building this is a little bit of an art, as Cython forces the use of the
compiler that built python itself (when linking). This mean ``mpicc`` cannot
simply be used, and we need to manually fetch the flags required for building
with MPI. This is likely to be the source of any difficulties building.

MPI Version (OpenMPI 1.8.2 or higher)
=====================================

Some of the features, especially distribution of matrice from global arrays
and files, make heavy use of advanced features of MPI, such as derived
datatypes and MPI-IO. Unfortunately many MPI distributions contain critical
bugs in these components (mostly due to ``ROMIO``), which means these will
fail in some common circumstances.

However, recent versions of OpenMPI contain a new implementation of MPI-IO
(called OMPIO) which seems to be issue free. This means that for full, and
successful usage you should try and use OpenMPI 1.8.2 or greater.
Additionally, you may need to force it to use OMPIO rather than ROMIO. This
can be done by calling with::

    $ mpirun -mca io ompio ...

or by setting the environment variable::

    $ export OMPI_MCA_io=ompio



 
