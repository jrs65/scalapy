from distutils.core import setup  
from distutils.extension import Extension  
from Cython.Distutils import build_ext  

import numpy as np

setup(  
   name = 'ScArray',  
   ext_modules=[ Extension('scarray', ['scarray.pyx', 'bcutil.c'], include_dirs=[np.get_include()]) ],  
   cmdclass = {'build_ext': build_ext}  
)
