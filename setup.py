from distutils.core import setup
import distutils.ccompiler
#from distutils.extension import Extension  
from Cython.Distutils import extension  
from Cython.Distutils import build_ext  

import subprocess
import os

## Remove CC variable which Intel compiler modulefiles keep setting
## as we must use the compiler that compiled python.
if 'CC' in os.environ:
    del os.environ['CC']

def runcommand(cmd):
    process = subprocess.Popen(cmd.split(), shell=False, stdout=subprocess.PIPE)    
    c = process.communicate()

    if process.returncode != 0:
        raise Exception("Something went wrong whilst running the command: %s" % cmd)

    return c[0]


mpilinkargs = runcommand('mpicc -showme:link').split()
mpicompileargs = runcommand('mpicc -showme:compile').split()


scl_lib = ['mkl_scalapack_lp64', 'mkl_rt', 'mkl_blacs_openmpi_lp64', 'iomp5', 'pthread']
scl_libdir = ['$(MKLROOT)/lib/intel64']

import numpy as np

setup(  
    name = 'PyScalapack',  
    ext_modules=[ extension.Extension('scarray', ['scarray.pyx', 'bcutil.c'],
                                      include_dirs=[np.get_include()],
                                      library_dirs=scl_libdir,
                                      libraries=scl_lib,
                                      extra_compile_args = (['-fopenmp'] + mpicompileargs),
                                      extra_link_args = (['-fopenmp'] + mpilinkargs)
                                      ),
                  extension.Extension('scroutine', ['scroutine.pyx'],
                                      include_dirs=[np.get_include()],
                                      library_dirs=scl_libdir,
                                      libraries=scl_lib,
                                      extra_compile_args = (['-fopenmp'] + mpicompileargs),
                                      extra_link_args = (['-fopenmp'] + mpilinkargs)
                                      )
                  ],  
    cmdclass = {'build_ext': build_ext}  
    )

