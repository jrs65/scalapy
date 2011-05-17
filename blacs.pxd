

## BLACS routines
cdef extern:
    void Cblacs_pinfo( int * mypnum, int * nprocs )
    void Cblacs_get( int icontxt,  int what, int * val)
    void Cblacs_gridinit( int * icontxt, char * order, int nprow, int npcol )
    void Cblacs_gridinfo( int icontxt, int * nprow, int * npcol, int * myprow, int * mypcol )


