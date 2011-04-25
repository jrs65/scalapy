#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"

#define AA(i,j) AA[(i)*M+(j)]



int main(int argc, char **argv) {
   int i, j, k;
/************  MPI ***************************/
   int myrank_mpi, nprocs_mpi;
   MPI_Init( &argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs_mpi);
   printf("%i %i\n", myrank_mpi, nprocs_mpi);
/************  BLACS ***************************/
   int ictxt, nprow, npcol, myrow, mycol,nb;
   int info,itemp;
   int ZERO=0,ONE=1;
   //nprow = 2; npcol = 2; nb =2;
   nprow = 1; npcol = 1; nb =2;
   Cblacs_pinfo( &myrank_mpi, &nprocs_mpi ) ;
   Cblacs_get( -1, 0, &ictxt );
   Cblacs_gridinit( &ictxt, "Row", nprow, npcol );
   Cblacs_gridinfo( ictxt, &nprow, &npcol, &myrow, &mycol );
   int M=5;
   double *AA = (double*) malloc(M*M*sizeof(double));
   for(i=0;i<M;i++ )
     for(j=0;j<M;j++)
        AA[i*M+j]=(2*i+3*j+1);

   double *X = (double*) malloc(M*sizeof(double));
   double xc[5];

   X[0]=1;X[1]=2;X[2]=3;X[3]=4;X[4]=5;

   int descA[9],descx[9],descy[9];
   int mA = numroc_( &M, &nb, &myrow, &ZERO, &nprow );
   int nA = numroc_( &M, &nb, &mycol, &ZERO, &npcol );
   int nx = numroc_( &M, &nb, &myrow, &ZERO, &nprow );
   int my = numroc_( &M, &nb, &myrow, &ZERO, &nprow );

   int tN = 100, tB = 5, tP = 2;
   //printf("%i  %i\n", numroc_( &tN, &tB, &myrow, &ZERO, &tP ), numrc(tN, tB, myrow, 0, tP)); 


   //printf("%i  %i\n", mA, numrc(M, nb, myrow, 0, nprow)); 
   descinit_(descA, &M,   &M,   &nb,  &nb,  &ZERO, &ZERO, &ictxt, &mA,  &info);
   descinit_(descx, &M, &ONE,   &nb, &ONE,  &ZERO, &ZERO, &ictxt, &nx, &info);

   descinit_(descy, &M, &ONE,   &nb, &ONE,  &ZERO, &ZERO, &ictxt, &my, &info);
   double *x = (double*) malloc(nx*sizeof(double));
   double *y = (double*) calloc(my,sizeof(double));
   double *A = (double*) malloc(mA*nA*sizeof(double));
   int sat,sut;
   for(i=0;i<mA;i++)

   for(j=0;j<nA;j++){
                sat= (myrow*nb)+i+(i/nb)*nb;
                sut= (mycol*nb)+j+(j/nb)*nb;
                A[j*mA+i]=AA(sat,sut);
        }

   //bc1_copy(X, xc, M, nb, nprow, myrow);

   for(i=0;i<nx;i++){
                sut= (myrow*nb)+i+(i/nb)*nb;
                x[i]=X[sut];
                printf("%f\n",xc[i]);
        }

   double alpha = 1.0; double beta = 0.0;
   pdgemv_("N",&M,&M,&alpha,A,&ONE,&ONE,descA,x,&ONE,&ONE,descx,&ONE,&beta,y,&ONE,&ONE,descy,&ONE);

   Cblacs_barrier(ictxt,"A");
   for(i=0;i<my;i++)
   printf("rank=%d %.2f \n", myrank_mpi,y[i]);
   Cblacs_gridexit( 0 );
   MPI_Finalize();
   return 0;
}
