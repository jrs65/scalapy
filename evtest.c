#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"

#include "bcutil.h"

#include <fcntl.h>
#include <unistd.h>

#include <errno.h>
#include <sys/stat.h>

#include <time.h>


int main(int argc, char **argv) {
  int Nr, Nc, Br, Bc, Pr, Pc;
  int pr, pc;
  if(argc < 9) {
    exit(-34);
  }
  Nr = atoi(argv[1]);
  Nc = atoi(argv[2]);
  Br = atoi(argv[3]);
  Bc = atoi(argv[4]);
  Pr = atoi(argv[5]);
  Pc = atoi(argv[6]);
/************  MPI ***************************/
   int myrank_mpi, nprocs_mpi;
   int info, ictxt;
   int ZERO=0,ONE=1;
   MPI_Init( &argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs_mpi);
/************  BLACS ***************************/

   //nprow = 2; npcol = 2; nb =2;
   //nprow = 2; npcol = 2; nb =2;
   Cblacs_pinfo( &myrank_mpi, &nprocs_mpi ) ;
   Cblacs_get( -1, 0, &ictxt );
   Cblacs_gridinit( &ictxt, "Row", Pr, Pc );
   Cblacs_gridinfo( ictxt, &Pr, &Pc, &pr, &pc );
   

   //scinit(argc, argv, &ictxt, &Pr, &Pc, &pr, &pc, &myrank_mpi, &nprocs_mpi);

   if(myrank_mpi == 0) {
     printf("Matrix size: %i x %i\n", Nr, Nc);
     printf("Block size:  %i x %i\n", Br, Bc);
     printf("Grid size:   %i x %i\n", Pr, Pc);
   }

   int nr, nc;

   nr = numrc(Nr, Br, pr, 0, Pr);
   nc = numrc(Nc, Bc, pc, 0, Pc);
   
   double *X = (double*) malloc(nr*nc*sizeof(double));
   double *evecs = (double*) malloc(nr*nc*sizeof(double));
   double *evals = (double*) malloc(Nr*sizeof(double));
   double * work;
   double wl;
   int lwork = -1;

   int * iwork;
   int liwork;

   time_t st, et;

   int fd;

   bc2d_mmap_load(argv[7], X, Nr, Nc, Br, Bc, Pr, Pc, pr, pc);
   
   Cblacs_barrier(ictxt,"B");
   st = time(NULL);

   int descX[9], desc_evecs[9];

   liwork = 7*Nr + 8*Pc + 2;
   iwork = (int *) malloc(sizeof(int) * liwork);

   //printf("%i  %i\n", mA, numrc(M, nb, myrow, 0, nprow)); 
   descinit_(descX,      &Nr, &Nc, &Br, &Bc,  &ZERO, &ZERO, &ictxt, &nr, &info);
   descinit_(desc_evecs, &Nr, &Nc, &Br, &Bc,  &ZERO, &ZERO, &ictxt, &nr, &info);
   //descinit_(desc_evals, &Nr, &ONE, &Br, &ONE,  &ZERO, &ZERO, &ictxt, &nr, &info);

   //pdgemv_("N",&M,&M,&alpha,A,&ONE,&ONE,descA,x,&ONE,&ONE,descx,&ONE,&beta,y,&ONE,&ONE,descy,&ONE);
   // Get workspace size
   pdsyevd_("V", "U", &Nr,
           X, &ONE, &ONE, descX,
           evals,
           evecs, &ONE, &ONE, desc_evecs, 
           &wl, &lwork, iwork, &liwork,
           &info);

   //printf("Required work array %f\n", wl);

   lwork = (int)wl;
   // Initialise work array
   work = (double *)malloc(sizeof(double) * lwork);
   
   // Compute.
   pdsyevd_("V", "U", &Nr,
           X, &ONE, &ONE, &descX,
           evals,
           evecs, &ONE, &ONE, desc_evecs, 
	    work, &lwork, iwork, &liwork,
           &info);

   //Cblacs_barrier(ictxt,"A");
   //for(i=0;i<my;i++)
   //printf("rank=%d %.2f \n", myrank_mpi,y[i]);

   if(pr == 0 && pc == 0) {
     fd = open(argv[8], O_WRONLY|O_CREAT, S_IREAD | S_IWRITE);
     if(write(fd, evals, Nr*sizeof(double)) == -1) {
       perror(NULL);
       exit(-30);
     }
     close(fd);


     fd = open(argv[9], O_WRONLY|O_CREAT, S_IREAD | S_IWRITE);
     ftruncate(fd, sizeof(double) * num_rpage(Nr, Br) * Nc);
     close(fd);
   }
   Cblacs_barrier(ictxt,"A");
   et = time(NULL);
   
   if(pr == 0 && pc == 0) {
     printf("Computation time: %f\n", (double)(et-st));
   }

   bc2d_mmap_save(argv[9], evecs, Nr, Nc, Br, Bc, Pr, Pc, pr, pc);

   Cblacs_gridexit( 0 );
   MPI_Finalize();
   return 0;
}
