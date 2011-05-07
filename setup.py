from distutils.core import setup
import distutils.ccompiler
from distutils.extension import Extension  
from Cython.Distutils import build_ext  

mpicc = distutils.ccompiler.new_compiler()

mpicc.set_executables(compiler = 'mpicc', linker_exe = 'mpicc')

import numpy as np
#-L$(MKLROOT)/lib/intel64 -lmkl_scalapack_lp64 -lmkl_rt -lmkl_blacs_intelmpi_lp64 -fopenmp -lpthread
setup(  
   name = 'ScArray',  
   ext_modules=[ Extension('scarray', ['scarray.pyx', 'bcutil.c'],
                           include_dirs=[np.get_include()],
                           #library_dirs=['$(MKLROOT)/lib/intel64'],
                           #libraries=['mkl_scalapack_lp64', 'mkl_rt', 'mkl_blacs_openmpi_lp64', 'pthread'],
                           extra_compile_args = ['-fopenmp'],
                           extra_link_args = ['-fopenmp'] ) ],  
   cmdclass = {'build_ext': build_ext}  
)
