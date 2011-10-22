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

  char *typeA = "N", *typeB = "N", *fname;
  
  int info, ZERO=0, ONE=1;

  struct timeval st, et;

  /* Initialising MPI stuff */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  printf("Process %i of %i.\n", rank, size);


  /* Parsing arguments */
  if(argc < 6) {
    exit(-3);
  }

  if(argc > 7) {
    typeA = argv[6];
    typeB = argv[7];
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
    printf("Operation type %s %s\n", typeA, typeB);
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

  Br = numroc_( &nside, &nblock, &ir, &ZERO, &nr);
  Bc = numroc_( &nside, &nblock, &ic, &ZERO, &nc);

  Cr = numroc_( &nside, &nblock, &ir, &ZERO, &nr);
  Cc = numroc_( &nside, &nblock, &ic, &ZERO, &nc);

  printf("Local array section %i x %i\n", Ar, Ac);

  /* Set descriptors */
  descinit_(descA, &nside, &nside, &nblock, &nblock, &ZERO, &ZERO, &ictxt, &Ar, &info);
  descinit_(descB, &nside, &nside, &nblock, &nblock, &ZERO, &ZERO, &ictxt, &Br, &info);
  descinit_(descC, &nside, &nside, &nblock, &nblock, &ZERO, &ZERO, &ictxt, &Cr, &info);

  /* Initialise and fill arrays */
  double *A = (double *)malloc(Ar*Ac*sizeof(double));
  double *B = (double *)malloc(Br*Bc*sizeof(double));
  double *C = (double *)malloc(Cr*Cc*sizeof(double));


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
  gettimeofday(&st, NULL);

  pdgemm_(typeA, typeB, &nside, &nside, &nside,
	  &alpha,
	  A, &ONE, &ONE, descA,
	  B, &ONE, &ONE, descB,
	  &beta,
	  C, &ONE, &ONE, descC );

  Cblacs_barrier(ictxt,"A");
  gettimeofday(&et, NULL);

  if(rank == 0) {
    double gfpc;
    double dt;
    FILE * fd;
    int ita = 0, itb = 0;
    if(! strcmp(typeA, "T")) ita = 1;
    if(! strcmp(typeB, "T")) itb = 1;

    dt = (double)((et.tv_sec-st.tv_sec) + (et.tv_usec-st.tv_usec)*1e-6);
    printf("=========\nTime taken: %g s\n", dt);

    gfpc = 2.0*pow(nside, 3) / (dt * 1e9 * ngrid * ngrid * nthread);

    printf("GFlops per core: %g\n=========\n", gfpc);

    fd = fopen(fname, "w");
    fprintf(fd, "%g %i %i %i %i %g %i %i\n", gfpc, nside, ngrid, nblock, nthread, dt, ita, itb);
    fclose(fd);
  }

  Cblacs_gridexit( 0 );
  MPI_Finalize();

}

