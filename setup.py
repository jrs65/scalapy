
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
    process = subprocess.Popen(cmd.split(), shell=False, stdout=subprocess.PIPE)
    c = process.communicate()

    if process.returncode != 0:
        raise Exception("Something went wrong whilst running the command: %s" % cmd)

    return c[0]


## Decide whether to use OpenMP or not
is_gcc = re.search('gcc', sysconfig.get_config_var('CC')) is not None
use_omp = is_gcc

## Try and decide whether to use Cython to compile the source or not.
try:
    from Cython.Distutils import build_ext
    import blagsjj
    HAVE_CYTHON = True

except ImportError as e:
    warnings.warn("Cython not installed.")
    #from numpy.distutils.command import build_ext

    HAVE_CYTHON = False


def cython_file(filename):
    filename = filename + ('.pyx' if HAVE_CYTHON else '.c')
    return filename


################# Configuration options to tweak. #########################

# Which mpi version? Should be either 'intelmpi' or 'openmpi'.
mpiversion = 'openmpi'

# Which ScaLapack version to use? Only 'intel' is supported at the moment.
scalapackversion = 'netlib'

############################################################################






## Make sure to remove CC variable which Intel compiler modulefiles keep setting
## as we must use the compiler that compiled python.

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


if scalapackversion == 'intel':
    # Set library includes (taking into account which MPI library we are using)."
    scl_lib = ['mkl_scalapack_lp64', 'mkl_rt', 'mkl_blacs_'+mpiversion+'_lp64', 'iomp5', 'pthread']
    scl_libdir = [os.environ['MKLROOT']+'/lib/intel64' if 'MKLROOT' in os.environ else '']
elif scalapackversion == 'netlib':
    scl_lib = ['scalapack', 'gfortran']
    scl_libdir = ['/usr/local/Cellar/gfortran/4.8.2/gfortran/lib']
else:
    raise Exception("Scalapack distribution unsupported. Please modify setup.py manually.")


use_omp = False
omp_args = ['-fopenmp'] if use_omp else []


mpi3_ext = Extension('scalapy.mpi3util', [cython_file('scalapy/mpi3util')],
                      include_dirs=['.', np.get_include(), mpi4py.get_include()],
                      extra_compile_args=mpicompileargs,
                      extra_link_args=mpilinkargs)

blacs_ext = Extension('scalapy.blacs', [cython_file('scalapy/blacs')],
                      include_dirs=['.', np.get_include(), mpi4py.get_include()],
                      library_dirs=scl_libdir, libraries=scl_lib,
                      extra_compile_args=mpicompileargs,
                      extra_link_args=mpilinkargs)

llpblas_ext = Extension('scalapy.lowlevel.pblas', ['scalapy/lowlevel/pblas.pyf'],
                        library_dirs=scl_libdir, libraries=scl_lib,
                        extra_compile_args=(mpicompileargs + omp_args),
                        extra_link_args=(mpilinkargs + omp_args))

llscalapack_ext = Extension('scalapy.lowlevel.scalapack', ['scalapy/lowlevel/scalapack.pyf'],
                            library_dirs=scl_libdir, libraries=scl_lib,
                            extra_compile_args=(mpicompileargs + omp_args),
                            extra_link_args=(mpilinkargs + omp_args))

setup(
    name='scalapy',
    packages=find_packages(),
    ext_modules=[mpi3_ext, blacs_ext, llpblas_ext, llscalapack_ext]
    #cmdclass={'build_ext': build_ext}
)

