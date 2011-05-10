from distutils.core import setup
import distutils.ccompiler
#from distutils.extension import Extension  
from Cython.Distutils import extension  
from Cython.Distutils import build_ext  

import subprocess

process = subprocess.Popen(['mpicc', '-showme:link'], shell=False, stdout=subprocess.PIPE)
mpilinkargs = process.communicate()[0].split()

process = subprocess.Popen(['mpicc', '-showme:compile'], shell=False, stdout=subprocess.PIPE)
mpicompileargs = process.communicate()[0].split()



import numpy as np

setup(  
    name = 'ScArray',  
    ext_modules=[ extension.Extension('scarray', ['scarray.pyx', 'bcutil.c'],
                                      include_dirs=[np.get_include(), '/usr/local/lib/python2.7/dist-packages/mpi4py/include'],
                                      library_dirs=['$(MKLROOT)/lib/intel64'],
                                      libraries=['mkl_scalapack_lp64', 'mkl_rt', 'mkl_blacs_openmpi_lp64', 'iomp5', 'pthread'],
                                      #libraries=['mkl_scalapack_lp64', 'mkl_blacs_openmpi_lp64', 'mkl_intel_lp64','mkl_gnu_thread', 'mkl_core', 'iomp5', 'pthread'],
                                      #libraries=['scalapack-openmpi'],
                                      extra_compile_args = (['-fopenmp'] + mpicompileargs),
                                      extra_link_args = (['-fopenmp'] + mpilinkargs)
                                      #extra_compile_args = mpicompileargs,
                                      #extra_link_args = mpilinkargs
                                      )
                  ],  
    cmdclass = {'build_ext': build_ext}  
    )
