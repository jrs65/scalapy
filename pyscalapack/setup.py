from distutils.core import setup
import distutils.ccompiler
from Cython.Distutils import extension  
from Cython.Distutils import build_ext  

import subprocess
import os
import numpy as np

def runcommand(cmd):
    process = subprocess.Popen(cmd.split(), shell=False, stdout=subprocess.PIPE)    
    c = process.communicate()

    if process.returncode != 0:
        raise Exception("Something went wrong whilst running the command: %s" % cmd)

    return c[0]




################# Configuration options to tweak. #########################

# Which mpi version? Should be either 'intelmpi' or 'openmpi'.
mpiversion = 'openmpi'

# Which ScaLapack version to use? Only 'intel' is supported at the moment.
scalapackversion = 'intel'

############################################################################






## Remove CC variable which Intel compiler modulefiles keep setting
## as we must use the compiler that compiled python.
if 'CC' in os.environ:
    del os.environ['CC']


if mpiversion == 'intelmpi':
    # Fetch command line, convert to a list, and remove the first item (the command).
    intelargs =  runcommand('mpicc -show').split()[1:]  
    mpilinkargs=intelargs
    mpicompileargs=intelargs
elif mpiversion == 'openmpi':
    # Fetch the arguments for linking and compiling.
    mpilinkargs = runcommand('mpicc -showme:link').split()
    mpicompileargs = runcommand('mpicc -showme:compile').split()
else:
    raise Exception("MPI library unsupported. Please modify setup.py manually.")


if scalapackversion == 'intel':
    # Set library includes (taking into account which MPI library we are using)."
    scl_lib = ['mkl_scalapack_lp64', 'mkl_rt', 'mkl_blacs_'+mpiversion+'_lp64', 'iomp5', 'pthread']
    scl_libdir = ['$(MKLROOT)/lib/intel64']
    
else:
    raise Exception("Scalapack distribution unsupported. Please modify setup.py manually.")


setup(  
    name = 'PyScalapack',
    ext_modules=[ extension.Extension('core', ['core.pyx', 'bcutil.c'],
                                      include_dirs=['.', np.get_include()],
                                      library_dirs=scl_libdir,
                                      libraries=scl_lib,
                                      extra_compile_args = (['-fopenmp'] + mpicompileargs),
                                      extra_link_args = (['-fopenmp'] + mpilinkargs)
                                      ),
                  extension.Extension('routines', ['routines.pyx'],
                                      include_dirs=['.', np.get_include()],
                                      library_dirs=scl_libdir,
                                      libraries=scl_lib,
                                      extra_compile_args = (['-fopenmp'] + mpicompileargs),
                                      extra_link_args = (['-fopenmp'] + mpilinkargs)
                                      )
                ],  
    cmdclass = {'build_ext': build_ext} )

