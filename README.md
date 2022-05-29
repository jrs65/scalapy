[![ci](https://github.com/pulkin/scalapy/actions/workflows/test.yml/badge.svg)](https://github.com/pulkin/scalapy/actions)


ScaLAPACK for Python (scalapy)
==============================

``scalapy`` is a wrapping of Scalapack such that it can be called by Python in
a friendly manner.

Operations are performed on ``DistributedMatrix`` objects which can be easily
created whilst hiding all the nasty details of block cyclic distribution.


Dependencies
------------

``scalapy`` supports both Python 2 and 3 (2.7, 3.2 or later).

The package depends upon two python packages ``numpy`` and ``mpi4py``. It is
written largely in pure Python, but some parts require ``Cython`` and ``f2py``.
It also requires an ``MPI`` distribution (OpenMPI and IntelMPI supported out the
box), and a ``Scalapack`` installation (both Intel MKL and NETLIB are currently
supported).

Installation
------------

To build just use the standard ``setup.py`` script.

    $ python setup.py install

It will attempt to probe you current environment to determine which MPI
distribution, and ScaLAPACK installation to use. As this isn't completely
robust, you can edit ``setup.py`` manually specify what to use.

Documentation
-------------

Limited, but improving, documentation is available ~~[here](http://jrs65.github.com/scalapy/)~~ (TBD).

MPI Version (OpenMPI 1.8.2 or higher)
-------------------------------------

Some of the features, especially distribution of matrices from global arrays and
files, make heavy use of advanced features of MPI, such as derived datatypes and
MPI-IO. Unfortunately many MPI distributions contain critical bugs in these
components (mostly due to ``ROMIO``), which means these will fail in some common
circumstances.

However, recent versions of OpenMPI contain a new implementation of MPI-IO
(called OMPIO) which seems to be issue free. This means that for full, and
successful usage you should try and use OpenMPI 1.8.2 or greater.
Additionally, you may need to force it to use OMPIO rather than ROMIO. This
can be done by calling with

    $ mpirun -mca io ompio ...

or by setting the environment variable

    $ export OMPI_MCA_io=ompio
