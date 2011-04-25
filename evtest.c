#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"

#define AA(i,j) AA[(i)*M+(j)]



int main(int argc, char **argv) {
  int Nr, Nc, Br, Bc, Pr, Pc;

  Nr = atoi(argv[0]);
  Nc = atoi(argv[1]);
  Br = atoi(argv[2]);
  Bc = atoi(argv[3]);
  Pr = atoi(argv[4]);
  Pc = atoi(argv[5]);

   int i, j, k;
/************  MPI ***************************/
   int myrank_mpi, nprocs_mpi;
   MPI_Init( &argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs_mpi);
   printf("%i %i\n", myrank_mpi, nprocs_mpi);
/************  BLACS ***************************/
   int info,itemp;
   int ZERO=0,ONE=1;
   //nprow = 2; npcol = 2; nb =2;
   nprow = 2; npcol = 2; nb =2;
   Cblacs_pinfo( &myrank_mpi, &nprocs_mpi ) ;
   Cblacs_get( -1, 0, &ictxt );
   Cblacs_gridinit( &ictxt, "Row", Pr, Pc );
   Cblacs_gridinfo( ictxt, &Pr, &Pc, &pr, &pc );

   int nr, nc;

   nr = numrc(Nr, Br, pr, 0, Pr);
   nc = numrc(Nc, Bc, pc, 0, Pc);
   
   double *X = (double*) malloc(nr*nc*sizeof(double));
   double *evecs = (double*) malloc(nr*nc*sizeof(double));
   double *evals = (double*) malloc(nr*sizeof(double));

   bc2d_mmap_load(argv[6], X, Nr, Nc, Br, Bc, Pr, Pc, pr, pc);

   int descX[9],desc_evecs[9],desc_evals[9];

   //printf("%i  %i\n", mA, numrc(M, nb, myrow, 0, nprow)); 
   descinit_(descX,      &Nr, &Nc, &Br, &Bc,  &ZERO, &ZERO, &ictxt, &nr, &info);
   descinit_(desc_evecs, &Nr, &Nc, &Br, &Bc,  &ZERO, &ZERO, &ictxt, &nr, &info);
   descinit_(desc_evals, &Nr, &ONE, &Br, &ONE,  &ZERO, &ZERO, &ictxt, &nr, &info);

   pdgemv_("N",&M,&M,&alpha,A,&ONE,&ONE,descA,x,&ONE,&ONE,descx,&ONE,&beta,y,&ONE,&ONE,descy,&ONE);

   Cblacs_barrier(ictxt,"A");
   for(i=0;i<my;i++)
   printf("rank=%d %.2f \n", myrank_mpi,y[i]);
   Cblacs_gridexit( 0 );
   MPI_Finalize();
   return 0;
}
