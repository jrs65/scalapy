
cimport numpy as np



cdef char * np_data(np.ndarray a)

    

cdef class DistributedMatrix(object):

    #cdef double * data

    cdef np.ndarray _local_array
    cdef np.ndarray _desc
    cdef object _context
    cdef object _dtype

    #Nr = 0
    #Nc = 0
    cdef readonly int Nr, Nc
    cdef readonly int Br, Bc

    cdef int * _getdesc(self)

    cdef char * _data(self)


