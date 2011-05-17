
cimport numpy as np



cdef void * np_data(np.ndarray a)

    

cdef class LocalVector(object):


    cdef np.ndarray _local_vector

    cdef np.ndarray _desc

    cdef object _context

    cdef readonly int N
    cdef readonly int B
    

    cdef int * _getdesc(self)

    cdef double * _data(self)




cdef class LocalMatrix(object):

    #cdef double * data

    cdef np.ndarray _local_matrix
    cdef np.ndarray _desc
    cdef object _context

    #Nr = 0
    #Nc = 0
    cdef readonly int Nr, Nc
    cdef readonly int Br, Bc

    cdef int * _getdesc(self)

    cdef double * _data(self)


