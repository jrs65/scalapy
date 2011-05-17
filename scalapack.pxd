

## Scalapack routines
cdef extern:
    void pdsyevd_( char * jobz, char * uplo, int * N,
                   double * A, int * ia, int * ja, int * desca,
                   double * w,
                   double * z, int * iz, int * jz, int * descz,
                   double * work, int * iwork, int * iwork, int * liwork,
                   int * info )

