#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"

#include <time.h>


int main(int argc, char **argv) {

  int ictxt, nside, ngrid, nblock, nthread;
  int rank, size;
  int ic, ir, nc, nr;

  int i, j;

  char *typeA = "N", *typeB = "N";
  
  int info, ZERO=0, ONE=1;

  time_t st, et;

  /* Initialising MPI stuff */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  printf("Process %i of %i.\n", rank, size);


  /* Parsing arguments */
  if(argc < 5) {
    exit(-3);
  }

  if(argc > 6) {
    typeA = argv[5];
    typeB = argv[6];
  }

  nside = atoi(argv[1]);
  ngrid = atoi(argv[2]);
  nblock = atoi(argv[3]);
  nthread = atoi(argv[4]);

  if(rank == 0) {
    printf("Multiplying matrices of size %i x %i\n", nside, nside);
    printf("Process grid size %i x %i\n", ngrid, ngrid);
    printf("Block size %i x %i\n", nblock, nblock);
    printf("Using %i OpenMP threads\n", nthread);
  }
  printf("Filling matrices.\n");

  /* Setting up BLACS */
  Cblacs_pinfo( &rank, &size ) ;
  Cblacs_get(-1, 0, &ictxt );
  Cblacs_gridinit(&ictxt, "Row", ngrid, ngrid);
  Cblacs_gridinfo(ictxt, &nr, &nc, &ir, &ic);

  int descA[9], descB[9], descC[9];

  /* Fetch local array sizes */
  int Ar, Ac, Br, Bc, Cr, Cc;

  printf("Filling matrices.\n");
  Ar = numroc_( &nside, &nblock, &ir, &ZERO, &nr);
  Ac = numroc_( &nside, &nblock, &ic, &ZERO, &nr);

  Br = numroc_( &nside, &nblock, &ir, &ZERO, &nr);
  Bc = numroc_( &nside, &nblock, &ic, &ZERO, &nr);

  Br = numroc_( &nside, &nblock, &ir, &ZERO, &nr);
  Bc = numroc_( &nside, &nblock, &ic, &ZERO, &nr);

  printf("Filling matrices.\n");
  /* Set descriptors */
  descinit_(descA, &nside, &nside, &nblock, &nblock, &ZERO, &ZERO, &ictxt, &Ar, &info);
  descinit_(descB, &nside, &nside, &nblock, &nblock, &ZERO, &ZERO, &ictxt, &Br, &info);
  descinit_(descC, &nside, &nside, &nblock, &nblock, &ZERO, &ZERO, &ictxt, &Cr, &info);

  printf("Filling matrices.\n");
  /* Initialise and fill arrays */
  double *A = (double *)malloc(Ar*Ac*sizeof(double));
  double *B = (double *)malloc(Br*Bc*sizeof(double));
  double *C = (double *)malloc(Cr*Cc*sizeof(double));

  printf("Filling matrices.\n");

  for(i = 0; i < Ar; i++) {
    for(j = 0; j < Ac; j++) {
      A[j*Ar + i] = drand48();
      B[j*Br + i] = drand48();
      C[j*Cr + i] = 0.0;
    }
  }

  double alpha = 1.0, beta = 0.0;

  if(rank == 0) {
    printf("Starting multiplication.\n");
  }


  Cblacs_barrier(ictxt,"A");
  st = time(NULL);

  pdgemm_(&typeA, &typeB, &nside, &nside,
	  &alpha,
	  A, &ONE, &ONE, descA,
	  B, &ONE, &ONE, descB,
	  &beta,
	  C, &ONE, &ONE, descC );

  Cblacs_barrier(ictxt,"A");
  et = time(NULL);

  if(rank == 0) {
    printf("=========\nTime taken: %g s\n=========\n", (et-st));
  }

  Cblacs_gridexit( 0 );
  MPI_Finalize();

}

