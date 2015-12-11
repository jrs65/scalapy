from __future__ import print_function

from setuptools import find_packages
from numpy.distutils.core import setup, Extension
from distutils import sysconfig

import subprocess
import os
import re
import warnings

import numpy as np
import mpi4py


def runcommand(cmd):
    process = subprocess.Popen(cmd.split(), shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    c = process.communicate()

    if process.returncode != 0:
        raise Exception("Something went wrong whilst running the command: %s" % cmd)

    return c[0]


def whichmpi():
    # Figure out which MPI environment this is
    import re
    mpiv = runcommand('mpirun -V')

    if re.search('Intel', mpiv):
        return 'intelmpi'
    elif re.search('Open MPI', mpiv):
        return 'openmpi'

    warnings.warn('Unknown MPI environment.')
    return None


def whichscalapack():
    # Figure out which Scalapack to use
    if 'MKLROOT' in os.environ:
        return 'intelmkl'
    else:
        return 'netlib'

## Decide whether to use OpenMP or not
is_gcc = re.search('gcc', sysconfig.get_config_var('CC')) is not None
use_omp = is_gcc

# Set the MPI version
mpiversion = whichmpi()

# Set the Scalapack version
scalapackversion = whichscalapack()

# Set to use OMP
use_omp = True


################# Configuration options to tweak. #########################
# Uncomment to override automatic detection and set manually

# Which mpi version? Should be either 'intelmpi' or 'openmpi'.
# mpiversion = 'openmpi'

# Which ScaLapack version to use? Only 'intel' is supported at the moment.
#scalapackversion = 'netlib'

# Use OMP routines or not?
# use_omp = True

############################################################################


## Find the MPI arguments required for building the modules.
if mpiversion == 'intelmpi':
    # Fetch command line, convert to a list, and remove the first item (the command).
    intelargs = runcommand('mpicc -show').split()[1:]
    mpilinkargs = intelargs
    mpicompileargs = intelargs
elif mpiversion == 'openmpi':
    # Fetch the arguments for linking and compiling.
    mpilinkargs = runcommand('mpicc -showme:link').split()
    mpicompileargs = runcommand('mpicc -showme:compile').split()
else:
    raise Exception("MPI library unsupported. Please modify setup.py manually.")


## Find the Scalapack library arguments required for building the modules.
if scalapackversion == 'intelmkl':
    # Set library includes (taking into account which MPI library we are using)."
    scl_lib = ['mkl_scalapack_lp64', 'mkl_rt', 'mkl_blacs_'+mpiversion+'_lp64', 'iomp5', 'pthread']
    scl_libdir = [os.environ['MKLROOT']+'/lib/intel64' if 'MKLROOT' in os.environ else '']
elif scalapackversion == 'netlib':
    scl_lib = ['scalapack', 'gfortran']
    scl_libdir = [ os.path.dirname(runcommand('gfortran -print-file-name=libgfortran.a')) ]
else:
    raise Exception("Scalapack distribution unsupported. Please modify setup.py manually.")


## Try and decide whether to use Cython to compile the source or not.
try:
    from Cython.Build import cythonize
    HAVE_CYTHON = True
except ImportError as e:
    warnings.warn("Cython not installed.")
    HAVE_CYTHON = False


def cython_file(filename):
    filename = filename + ('.pyx' if HAVE_CYTHON else '.c')
    return filename


omp_args = ['-fopenmp'] if use_omp else []

print("=============================================================================")
print("Building Scalapy....")
print()
print("  ScaLAPACK: %s" % scalapackversion)
print("  MPI: %s" % mpiversion)
print("  OpenMP: %s" % repr(use_omp))
print()
print("  Compile args: %s" % repr(mpicompileargs))
print("  Libraries: %s" % repr(scl_lib + mpilinkargs))
print("  Library path: %s" % repr(scl_libdir))
print()
print("=============================================================================")

## Setup the extensions we are going to build
mpi3_ext = Extension('scalapy.mpi3util', [cython_file('scalapy/mpi3util')],
                     include_dirs=['.', np.get_include(), mpi4py.get_include()],
                     extra_compile_args=mpicompileargs,
                     extra_link_args=mpilinkargs)

blacs_ext = Extension('scalapy.blacs', [cython_file('scalapy/blacs')],
                      include_dirs=['.', np.get_include(), mpi4py.get_include()],
                      library_dirs=scl_libdir, libraries=scl_lib,
                      extra_compile_args=mpicompileargs,
                      extra_link_args=mpilinkargs)

llredist_ext = Extension('scalapy.lowlevel.redist', ['scalapy/lowlevel/redist.pyf'],
                         library_dirs=scl_libdir, libraries=scl_lib,
                         extra_compile_args=(mpicompileargs + omp_args),
                         extra_link_args=(mpilinkargs + omp_args))

llpblas_ext = Extension('scalapy.lowlevel.pblas', ['scalapy/lowlevel/pblas.pyf'],
                        library_dirs=scl_libdir, libraries=scl_lib,
                        extra_compile_args=(mpicompileargs + omp_args),
                        extra_link_args=(mpilinkargs + omp_args))

llscalapack_ext = Extension('scalapy.lowlevel.scalapack', ['scalapy/lowlevel/scalapack.pyf'],
                            library_dirs=scl_libdir, libraries=scl_lib,
                            extra_compile_args=(mpicompileargs + omp_args),
                            extra_link_args=(mpilinkargs + omp_args))

## Apply Cython to the extensions if it's installed
exts = [mpi3_ext, blacs_ext, llpblas_ext, llscalapack_ext, llredist_ext]
if HAVE_CYTHON:
    exts = cythonize(exts, include_path=['.', np.get_include(), mpi4py.get_include()])

setup(
    name='scalapy',
    author='J. Richard Shaw',
    description='Python bindings for ScaLAPACK.',
    url='http://github.com/jrs65/scalapy/',
    license='BSD',
    packages=find_packages(),
    ext_modules=exts
)
