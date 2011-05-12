
import os.path
import sys
import numpy as np
cimport numpy as np
import sys

from mpi4py import MPI

#from libc.stdlib import size_t

ctypedef unsigned long size_t

cdef extern from "stdlib.h":
    void free(void* ptr)
    void* malloc(size_t size)

cdef extern from "bcutil.h":
    int bc1d_copy_pagealign(double * src, double * dest, int N, int B)
    int bc2d_copy_pagealign(double * src, double * dest, int Nr, int Nc, int Br, int Bc)
    int bc1d_from_pagealign(double * src, double *dest, size_t N, size_t B)
    int bc2d_from_pagealign(double * src, double * dest, size_t Nr, size_t Nc, size_t Br, size_t Bc)
    int num_rpage(int N, int B)

    size_t numrc(size_t N, size_t B, size_t p, size_t p0, size_t P)
    int bc1d_mmap_load(char * file, double * dest, size_t N, size_t B, size_t P, size_t p)
    int bc2d_mmap_load(char * file, double * dest, size_t Nr, size_t Nc, size_t Br, size_t Bc, size_t Pr, size_t Pc, size_t pr, size_t pc)

    int bc1d_mmap_save(char * file, double * src, size_t N, size_t B, size_t P, size_t p)
    int bc2d_mmap_save(char * file, double * src, size_t Nr, size_t Nc, size_t Br, size_t Bc, size_t Pr, size_t Pc, size_t pr, size_t pc)

    
    int bc1d_copy_forward(double * src, double *dest, size_t N, size_t B, size_t P, size_t p)
    int bc2d_copy_forward(double * src, double * dest, size_t Nr, size_t Nc, size_t Br, size_t Bc, size_t Pr, size_t Pc, size_t pr, size_t pc)


    int bc1d_copy_backward(double * src, double *dest, size_t N, size_t B, size_t P, size_t p)
    int bc2d_copy_backward(double * src, double * dest, size_t Nr, size_t Nc, size_t Br, size_t Bc, size_t Pr, size_t Pc, size_t pr, size_t pc)

    int scinit(int argc, char ** argv, int * ictxt, int * Pr, int * Pc, int * pr, int * pr, int * rank, int * size)


cdef extern:
    void Cblacs_pinfo( int * mypnum, int * nprocs )
    void Cblacs_get( int icontxt,  int what, int * val)
    void Cblacs_gridinit( int * icontxt, char * order, int nprow, int npcol )
    void Cblacs_gridinfo( int icontxt, int * nprow, int * npcol, int * myprow, int * mypcol )


_context = None

def initmpi():
    global _context
    
    cdef int pnum, nprocs, ictxt, row, col, nrows, ncols
    cdef int rank, size
    comm = MPI.COMM_WORLD
    ct = ProcessContext()
    ct.rank = comm.Get_rank()
    ct.size = comm.Get_size()
    print "MPI: %i of %i" % (ct.rank, ct.size)
    
    Cblacs_pinfo(&pnum, &nprocs)
    print "BLACS pinfo %i %i" % (pnum, nprocs)

    ## Figure out what to do when we have spare MPI procs
    side = int((nprocs*1.0)**0.5)

    Cblacs_get(-1, 0, &ictxt)
    print "BLACS context: %i" % ictxt
    
    Cblacs_gridinit(&ictxt, "Row", side, side)
    Cblacs_gridinfo(ictxt, &nrows, &ncols, &row, &col)

    ct.num_rows = nrows
    ct.num_cols = ncols

    ct.row = row
    ct.col = col
    print "MPI %i: position (%i,%i) in %i x %i" % (ct.rank, ct.row, ct.col, ct.num_rows, ct.num_cols)

    _context = ct



    

    
class ProcessContext(object):
    r"""The position in the process grid."""
    num_rows = 1
    num_cols = 1

    row = 0
    col = 0

    mpi_rank = 0
    mpi_size = 1


def vector_pagealign(vec, blocksize):
    r"""Page aligns the blocks in a matrix, and makes Fortran ordered.

    Parameters
    ==========
    vec : ndarray
        The vector to page align (can either by C or Fortran ordered).
    blocksize : scalar
        The blocksize of the vector.

    Returns
    =======
    v2 : ndarray
        The page aligned vector.
    """
    cdef np.ndarray[np.float64_t, ndim=1] v1
    cdef np.ndarray[np.float64_t, ndim=1] v2

    v1 = np.ascontiguousarray(vec)

    N = vec.shape[0]
    B = blocksize

    nr = num_rpage(N, B)

    v2 = np.empty(nr, order='F')

    bc1d_copy_pagealign(<double *>v1.data, <double *>v2.data, N, B)

    return v2


def matrix_pagealign(mat, blocksize):
    r"""Page aligns the blocks in a matrix, and makes Fortran ordered.

    Parameters
    ==========
    mat : ndarray
        The matrix to page align (can either by C or Fortran ordered).
    blocksize : array_like
        The blocksize, the first and second elements correspond to the
        row and column blocks respectively.

    Returns
    =======
    m2 : ndarray
        The page aligned matrix.
    """
    cdef np.ndarray[np.float64_t, ndim=2] m1
    cdef np.ndarray[np.float64_t, ndim=2] m2

    m1 = np.asfortranarray(mat)

    Nr, Nc = mat.shape
    Br, Bc = blocksize

    nr = num_rpage(Nr, Br)

    m2 = np.empty((nr, Nc), order='F')

    bc2d_copy_pagealign(<double *>m1.data, <double *>m2.data, Nr, Nc, Br, Bc)

    return m2


def vector_from_pagealign(vecp, size, blocksize):
    r"""Page aligns the blocks in a matrix, and makes Fortran ordered.

    Parameters
    ==========
    vecp : ndarray
        The vector to page align (can either by C or Fortran ordered).
    blocksize : scalar
        The blocksize of the vector.

    Returns
    =======
    v2 : ndarray
        The page aligned vector.
    """
    cdef np.ndarray[np.float64_t, ndim=1] v1
    cdef np.ndarray[np.float64_t, ndim=1] v2

    v1 = vecp

    N = size
    B = blocksize

    nr = num_rpage(N, B)

    if len(v1) < nr:
        raise Exception("Source vector not long enough.")

    v2 = np.empty(N, order='F')

    bc1d_from_pagealign(<double *>v1.data, <double *>v2.data, N, B)

    return v2


def matrix_from_pagealign(matp, size, blocksize):
    r"""Page aligns the blocks in a matrix, and makes Fortran ordered.

    Parameters
    ==========
    matp : ndarray
        The matrix to page align (can either by C or Fortran ordered).
    blocksize : array_like
        The blocksize, the first and second elements correspond to the
        row and column blocks respectively.

    Returns
    =======
    m2 : ndarray
        The page aligned matrix.
    """
    cdef np.ndarray[np.float64_t, ndim=1] m1
    cdef np.ndarray[np.float64_t, ndim=2] m2

    #m1 = matp.flatten()
    m1 = matp.reshape(-1, order='A')

    Nr, Nc = size
    Br, Bc = blocksize

    nr = num_rpage(Nr, Br)

    if np.size(m1) < nr*Nc:
        raise Exception("Source matrix not long enough. Is %i x %i, should be %i x %i." % (matp.shape[0], matp.shape[1], nr, Nc))

    m2 = np.empty((Nr, Nc), order='F')

    bc2d_from_pagealign(<double *>m1.data, <double *>m2.data, Nr, Nc, Br, Bc)

    return m2

    

cdef class ScVector(object):

    cdef double * data

    local_vector = None
    global_vector = None

    def __init__(self, N, B, fname = None, context = None):
        cdef np.ndarray[np.float64_t, ndim=1] vc
        
        self.N = N
        self.B = B

        if not context:
            if not _context:
                raise Exception("No supplied or default context.")
            else:
                self.context = _context
        else:
            self.context = context

        n = numrc(self.N, self.B, self.context.row, 0, self.context.num_rows)
        self.local_vector = np.empty(n)
        
        if fname:
            if os.path.exists(fname):

                # Check size

                vc = self.local_vector
                bc1d_mmap_load(fname, <double *>vc.data, self.N, self.B, 
                               self.context.num_rows, self.context.row)
                

    def to_file(fname):
        pass


class ScMatrix(object):

    #cdef double * data

    local_matrix = None
    global_matrix = None

    def __init__(self, Nr, Nc, Br, Bc, context = None):
        self.Nr = Nr
        self.Br = Br
        self.Nc = Nc
        self.Bc = Bc
        if not context:
            if not _context:
                raise Exception("No supplied or default context.")
            else:
                self.context = _context
        else:
            self.context = context

        nr = numrc(self.Nr, self.Br, self.context.row, 0, self.context.num_rows)
        nc = numrc(self.Nc, self.Bc, self.context.col, 0, self.context.num_cols)
        self.local_matrix = np.empty((nr,nc), order='F')

    @classmethod
    def fromfile(cls, file, Nr, Nc, Br, Bc):
        cdef np.ndarray[np.float64_t, ndim=2] mc

        m = cls(Nr, Nc, Br, Bc)
        
        if os.path.exists(file):
            # Check size
            mc = m.local_matrix
            bc2d_mmap_load(file, <double *>mc.data, m.Nr, m.Nc, m.Br, m.Bc, 
                           m.context.num_rows, m.context.num_cols, 
                           m.context.row, m.context.col)

        return m

    @classmethod
    def fromarray(cls, array, Br, Bc):
        cdef np.ndarray[np.float64_t, ndim=2] mc
        cdef np.ndarray[np.float64_t, ndim=2] ac

        Nr = array.shape[0]
        Nc = array.shape[1]

        m = cls(Nr, Nc, Br, Bc)
        
        # Check size
        mc = m.local_matrix
        ac = np.asfortranarray(array)
        bc2d_copy_forward(<double *>ac.data, <double *>mc.data, m.Nr, m.Nc, m.Br, m.Bc, 
                          m.context.num_rows, m.context.num_cols, 
                          m.context.row, m.context.col)

        return m
            
                


    def to_file(fname):
        pass
    
