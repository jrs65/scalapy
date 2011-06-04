

## Scalapack routines
cdef extern:
    void pdsyevd_( char * jobz, char * uplo, int * N,
                   double * A, int * ia, int * ja, int * desca,
                   double * w,
                   double * z, int * iz, int * jz, int * descz,
                   double * work, int * iwork, int * iwork, int * liwork,
                   int * info )


    void pdpotrf_( char * uplo, int * N,
                   double * A, int * ia, int * ja, int * desca,
                   int * info )

    void pdgemm_( char * transa, char * transb, int * m, int * n, int * k, double * alpha,
                  double * A, int * ia, int * ja, int * desca,
                  double * B, int * ib, int * jb, int * descb,
                  double * beta,
                  double * C, int * ic, int * jc, int * descc )
