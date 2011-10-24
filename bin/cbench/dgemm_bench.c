#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include <sys/time.h>

#ifdef _OPENMP
#include <omp.h>
#endif


int main(int argc, char **argv) {

  int nside, nthread;

  int i, j;

  char *fname;
  
  int info, ZERO=0, ONE=1;

  struct timeval st, et;

  double dt_nn, dt_nt, dt_tn, dt_tt;
  double gfpc_nn, gfpc_nt, gfpc_tn, gfpc_tt;

  /* Parsing arguments */
  if(argc < 3) {
    exit(-3);
  }

  nside = atoi(argv[1]);
  nthread = atoi(argv[2]);
  fname = argv[3];

  printf("Multiplying matrices of size %i x %i\n", nside, nside);
  printf("Using %i OpenMP threads\n", nthread);
  



  #ifdef _OPENMP
  printf("Setting OMP_NUM_THREADS=%i\n", nthread);
  omp_set_num_threads(nthread);
  #endif

  /* Initialise and fill arrays */
  double *A = (double *)malloc(nside*nside*sizeof(double));
  double *B = (double *)malloc(nside*nside*sizeof(double));
  double *C = (double *)malloc(nside*nside*sizeof(double));


  for(i = 0; i < nside; i++) {
    for(j = 0; j < nside; j++) {
      A[j*nside + i] = drand48();
      B[j*nside + i] = drand48();
      C[j*nside + i] = 0.0;
    }
  }

  double alpha = 1.0, beta = 0.0;

  //========================
  
  printf("Starting multiplication (NN).\n");

  gettimeofday(&st, NULL);

  dgemm_("N", "N", &nside, &nside, &nside,
	  &alpha, A, &nside, B, &nside, &beta, C, &nside );

  gettimeofday(&et, NULL);
  dt_nn = (double)((et.tv_sec-st.tv_sec) + (et.tv_usec-st.tv_usec)*1e-6);
  gfpc_nn = 2.0*pow(nside, 3.0) / (dt_nn * 1e9 * nthread);

  printf("Done.\n=========\nTime taken: %g s\nGFlops per core: %g\n=========\n", dt_nn, gfpc_nn);

  //========================



  //========================
  
 printf("Starting multiplication (NT).\n");

  gettimeofday(&st, NULL);

  dgemm_("N", "T", &nside, &nside, &nside,
	  &alpha, A, &nside, B, &nside, &beta, C, &nside );

  gettimeofday(&et, NULL);
  dt_nt = (double)((et.tv_sec-st.tv_sec) + (et.tv_usec-st.tv_usec)*1e-6);
  gfpc_nt = 2.0*pow(nside, 3.0) / (dt_nt * 1e9 * nthread);

 printf("Done.\n=========\nTime taken: %g s\nGFlops per core: %g\n=========\n", dt_nn, gfpc_nn);

  //========================



  //========================
 
  printf("Starting multiplication (TN).\n");

  gettimeofday(&st, NULL);

  dgemm_("T", "N", &nside, &nside, &nside,
	  &alpha, A, &nside, B, &nside, &beta, C, &nside );

  gettimeofday(&et, NULL);
  dt_tn = (double)((et.tv_sec-st.tv_sec) + (et.tv_usec-st.tv_usec)*1e-6);
  gfpc_tn = 2.0*pow(nside, 3.0) / (dt_tn * 1e9 * nthread);

  printf("Done.\n=========\nTime taken: %g s\nGFlops per core: %g\n=========\n", dt_tn, gfpc_tn);

  //========================



  //========================
  
  printf("Starting multiplication (TT).\n");

  gettimeofday(&st, NULL);

  dgemm_("T", "T", &nside, &nside, &nside,
	  &alpha, A, &nside, B, &nside, &beta, C, &nside );

  gettimeofday(&et, NULL);
  dt_tt = (double)((et.tv_sec-st.tv_sec) + (et.tv_usec-st.tv_usec)*1e-6);
  gfpc_tt = 2.0*pow(nside, 3.0) / (dt_tt * 1e9 * nthread);

  printf("Done.\n=========\nTime taken: %g s\nGFlops per core: %g\n=========\n", dt_tt, gfpc_tt);

  //========================






  FILE * fd;
  fd = fopen(fname, "w");
  fprintf(fd, "%g %g %g %g %i %i %i %i %g %g %g %g\n", gfpc_nn, gfpc_nt, gfpc_tn, gfpc_tt, nside, 1, 1, nthread, dt_nn, dt_nt, dt_tn, dt_tt);
  fclose(fd);

}

