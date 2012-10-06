#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"
#include <math.h>

#include <sys/time.h>

#ifdef _OPENMP
#include <omp.h>
#endif


int main(int argc, char **argv) {

  int ictxt, nside, ngrid, nblock, nthread;
  int rank, size;
  int ic, ir, nc, nr;

  int i, j;

  char *fname;
  
  int info, ZERO=0, ONE=1;

  struct timeval st, et;

  double dtnn, dtnt, dttn, dttt;
  double gfpc_nn, gfpc_nt, gfpc_tn, gfpc_tt;

  /* Initialising MPI stuff */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  printf("Process %i of %i.\n", rank, size);


  /* Parsing arguments */
  if(argc < 6) {
    exit(-3);
  }

  nside = atoi(argv[1]);
  ngrid = atoi(argv[2]);
  nblock = atoi(argv[3]);
  nthread = atoi(argv[4]);
  fname = argv[5];

  if(rank == 0) {
    printf("Multiplying matrices of size %i x %i\n", nside, nside);
    printf("Process grid size %i x %i\n", ngrid, ngrid);
    printf("Block size %i x %i\n", nblock, nblock);
    printf("Using %i OpenMP threads\n", nthread);
  }



  #ifdef _OPENMP
  if(rank == 0) printf("Setting OMP_NUM_THREADS=%i\n", nthread);
  omp_set_num_threads(nthread);
  #endif

  /* Setting up BLACS */
  Cblacs_pinfo( &rank, &size ) ;
  Cblacs_get(-1, 0, &ictxt );
  Cblacs_gridinit(&ictxt, "Row", ngrid, ngrid);
  Cblacs_gridinfo(ictxt, &nr, &nc, &ir, &ic);

  int descA[9], descB[9], descC[9];

  /* Fetch local array sizes */
  int Ar, Ac, Br, Bc, Cr, Cc;

  Ar = numroc_( &nside, &nblock, &ir, &ZERO, &nr);
  Ac = numroc_( &nside, &nblock, &ic, &ZERO, &nc);

  printf("Local array section %i x %i\n", Ar, Ac);

  /* Set descriptors */
  descinit_(descA, &nside, &nside, &nblock, &nblock, &ZERO, &ZERO, &ictxt, &Ar, &info);
  descinit_(descev, &nside, &nside, &nblock, &nblock, &ZERO, &ZERO, &ictxt, &Ar, &info);  

  /* Initialise and fill arrays */
  double *A = (double *)malloc(Ar*Ac*sizeof(double));
  double *evecs = (double *)malloc(Ar*Ac*sizeof(double));

  double *evals = (double *)malloc(Ar * sizeof(double));

  for(i = 0; i < Ar; i++) {
    for(j = 0; j < Ac; j++) {
      A[j*Ar + i] = drand48();
      evecs[j*Ar + i] = 0.0;
    }
  }

  int liwork = 7*A.Nr + 8*A.context.num_cols + 2;
  int * iwork = <int *>malloc(sizeof(int) * liwork);


  char * uplo = "U";
  double tlw;

  // Workspace size inquiry
  int lwork = -1
  pdsyevd_("V", uplo, &Ar,
             A, &_ONE, &_ONE, descA,
             evals,
             evecs, &_ONE, &_ONE, descev,
             &tlw, &lwork, iwork, &liwork,
             &info);
    
  // Initialise workspace to correct length
  lwork =(int)tlw;
  work = (double *)malloc(sizeof(double) * lwork);

  // Compute eigen problem
  pdsyevd_("V", uplo, &Ar,
           A, &_ONE, &_ONE, descA,
           evals,
           evecs, &_ONE, &_ONE, descev,
           work, &lwork, iwork, &liwork,
           &info);

  free(iwork);
  free(work);

  double alpha = 1.0, beta = 0.0;

  //========================
  
  if(rank == 0) printf("Starting multiplication (NN).\n");

  Cblacs_barrier(ictxt,"A");
  gettimeofday(&st, NULL);

  pdgemm_("N", "N", &nside, &nside, &nside,
	  &alpha,
	  A, &ONE, &ONE, descA,
	  B, &ONE, &ONE, descB,
	  &beta,
	  C, &ONE, &ONE, descC );

  Cblacs_barrier(ictxt,"A");
  gettimeofday(&et, NULL);
  dtnn = (double)((et.tv_sec-st.tv_sec) + (et.tv_usec-st.tv_usec)*1e-6);
  gfpc_nn = 2.0*pow(nside, 3) / (dtnn * 1e9 * ngrid * ngrid * nthread);

  if(rank == 0) printf("Done.\n=========\nTime taken: %g s\nGFlops per core: %g\n=========\n", dtnn, gfpc_nn);

  //========================



  //========================

  if(rank == 0) printf("Starting multiplication (NT).\n");

  Cblacs_barrier(ictxt,"A");
  gettimeofday(&st, NULL);

  pdgemm_("N", "T", &nside, &nside, &nside,
	  &alpha,
	  A, &ONE, &ONE, descA,
	  B, &ONE, &ONE, descB,
	  &beta,
	  C, &ONE, &ONE, descC );

  Cblacs_barrier(ictxt,"A");
  gettimeofday(&et, NULL);
  dtnt = (double)((et.tv_sec-st.tv_sec) + (et.tv_usec-st.tv_usec)*1e-6);
  gfpc_nt = 2.0*pow(nside, 3) / (dtnt * 1e9 * ngrid * ngrid * nthread);

  if(rank == 0) printf("Done.\n=========\nTime taken: %g s\nGFlops per core: %g\n=========\n", dtnt, gfpc_nt);

  //========================



  //========================

  if(rank == 0) printf("Starting multiplication (TN).\n");

  Cblacs_barrier(ictxt,"A");
  gettimeofday(&st, NULL);

  pdgemm_("T", "N", &nside, &nside, &nside,
	  &alpha,
	  A, &ONE, &ONE, descA,
	  B, &ONE, &ONE, descB,
	  &beta,
	  C, &ONE, &ONE, descC );

  Cblacs_barrier(ictxt,"A");
  gettimeofday(&et, NULL);
  dttn = (double)((et.tv_sec-st.tv_sec) + (et.tv_usec-st.tv_usec)*1e-6);
  gfpc_tn = 2.0*pow(nside, 3) / (dttn * 1e9 * ngrid * ngrid * nthread);

  if(rank == 0) printf("Done.\n=========\nTime taken: %g s\nGFlops per core: %g\n=========\n", dttn, gfpc_tn);

  //========================



  //========================

  if(rank == 0) printf("Starting multiplication (TT).\n");

  Cblacs_barrier(ictxt,"A");
  gettimeofday(&st, NULL);

  pdgemm_("T", "T", &nside, &nside, &nside,
	  &alpha,
	  A, &ONE, &ONE, descA,
	  B, &ONE, &ONE, descB,
	  &beta,
	  C, &ONE, &ONE, descC );

  Cblacs_barrier(ictxt,"A");
  gettimeofday(&et, NULL);
  dttt = (double)((et.tv_sec-st.tv_sec) + (et.tv_usec-st.tv_usec)*1e-6);
  gfpc_tt = 2.0*pow(nside, 3) / (dttt * 1e9 * ngrid * ngrid * nthread);

  if(rank == 0) printf("Done.\n=========\nTime taken: %g s\nGFlops per core: %g\n=========\n", dttt, gfpc_tt);

  //========================




  if(rank == 0) {
    FILE * fd;
    fd = fopen(fname, "w");
    fprintf(fd, "%g %g %g %g %i %i %i %i %g %g %g %g\n", gfpc_nn, gfpc_nt, gfpc_tn, gfpc_tt, nside, ngrid, nblock, nthread, dtnn, dtnt, dttn, dttt);
    fclose(fd);
  }

  Cblacs_gridexit( 0 );
  MPI_Finalize();

}

