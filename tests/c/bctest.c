#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <fcntl.h>
#include <unistd.h>

#include <sys/mman.h>

#include <errno.h>

#include "bcutil.h"



int main(int argc, char **argv) {
  double * X; 
  double * xc;

  double * X2r, *X2c;
  double * xc2, *xr2;

  double * X2r2, * xr22;

  int i, j, nr, nc;

  int N = 19, B = 3, P = 3, p = 0;

  int stride = 5;
  



  int Nc = 17, Nr = 5, Bc = 3, Br = 2, Pc = 3, Pr = 2, pc = 2, pr = 0;

  int pi, pj;

  int fd, fd1, fd2;

  int m2an;

  double * X2m;

  double * Xs, * xsc, * Xs2;

  printf("\n1d test\n");

  X = (double *)malloc(sizeof(double) * N);

  for(i = 0; i < N; i++) {
    X[i] = i;
  }
  
  nr = numrc(N, B, p, 0, P);
  xc = (double *)malloc(sizeof(double) * nr);

  printf("Local array size: %i\n", nr);

  bc1d_copy_forward(X, xc, N, B, P, p);

  for(i = 0; i < nr; i++) {
    
    printf("%10i %10.2f %10.2f\n", i, xc[i], X[i % B + B * (p + P * (i / B))]);

  }




  printf("\n1d test\n");


  X2r = (double *)malloc(sizeof(double) * Nr * Nc);
  X2r2 = (double *)malloc(sizeof(double) * Nr * Nc);
  X2c = (double *)malloc(sizeof(double) * Nr * Nc);

  for(i = 0; i < Nc; i++) {
    for(j = 0; j < Nr; j++) {
      X2c[i*Nr + j] = i;
      X2r[i*Nr + j] = j;
    }
  }
  
  nr = numrc(Nr, Br, pr, 0, Pr);
  nc = numrc(Nc, Bc, pc, 0, Pc);
  xc2 = (double *)malloc(sizeof(double) * nr * nc);
  xr2 = (double *)malloc(sizeof(double) * nr * nc);

  printf("Local array size: %i x %i\n", nc, nr);

  bc2d_copy_forward(X2c, xc2, Nr, Nc, Br, Bc, Pr, Pc, pr, pc);
  bc2d_copy_forward(X2r, xr2, Nr, Nc, Br, Bc, Pr, Pc, pr, pc);

  for(i = 0; i < nc; i++) {
    for(j = 0; j < nr; j++) {
      //printf("%10i %10.2f %10.2f\n", i, xc[i], X[i % B + B * (p + P * (i / B))]);
      printf("%5.1f  ", xc2[i*nr+j]);
    }
    printf("\n");
  }
  printf("\n\n");
  for(i = 0; i < nc; i++) {
    for(j = 0; j < nr; j++) {
      //printf("%10i %10.2f %10.2f\n", i, xc[i], X[i % B + B * (p + P * (i / B))]);
      printf("%5.1f  ", xr2[i*nr+j]);
    }
    printf("\n");
  }


  printf("\n\n");
  for(pi = 0; pi < Pc; pi++) {
    for(pj = 0; pj < Pr; pj++) {
      nc = numrc(Nc, Bc, pi, 0, Pc);
      nr = numrc(Nr, Br, pj, 0, Pr);
      xr22 = (double *)malloc(sizeof(double) * nc * nr);

      bc2d_copy_forward(X2r, xr22, Nr, Nc, Br, Bc, Pr, Pc, pj, pi);
      bc2d_copy_backward(xr22, X2r2, Nr, Nc, Br, Bc, Pr, Pc, pj, pi);

      free(xr22);
    }
  }

  for(i = 0; i < Nc; i++) {
    for(j = 0; j < Nr; j++) {
      printf("%5.1f  ", X2r[i*Nr+j]);
    }
    printf("\n");
  }

  printf("\n\n");

  for(i = 0; i < Nc; i++) {
    for(j = 0; j < Nr; j++) {
      printf("%5.1f  ", X2r2[i*Nr+j]);
    }
    printf("\n");
  }
  
  printf("\n\n");

  printf("Page size: %i\n", getpagesize());

  fd = open("testfile1.dat", O_WRONLY|O_CREAT, S_IREAD | S_IWRITE);
  if(write(fd, X2r, sizeof(double)*Nr*Nc) == -1) {
    printf("Error.\n");
    perror(NULL);
    exit(-30);
    }
  if(close(fd) != 0) {
    perror(NULL);
    exit(-223);
  }

  fd1 = open("testfile1.dat", O_RDONLY);
  
  nr = numrc(Nr, Br, pr, 0, Pr);
  nc = numrc(Nc, Bc, pc, 0, Pc);

  X2m = (double *)mmap(NULL, sizeof(double) * Nr* Nc, PROT_READ, MAP_PRIVATE, fd1, 0);
  if((long)X2m == -1) { perror(NULL); exit(-226); }

  bc2d_copy_forward(X2m, xc2, Nr, Nc, Br, Bc, Pr, Pc, pr, pc);

  for(i = 0; i < nc; i++) {
    for(j = 0; j < nr; j++) {
      //printf("%10i %10.2f %10.2f\n", i, xc[i], X[i % B + B * (p + P * (i / B))]);
      printf("%5.1f  ", xc2[i*nr+j]);
    }
    printf("\n");
  }

  munmap(X2m, sizeof(double) * Nr *Nc);
  close(fd1);

  printf("\n\n");

  Xs = (double *)calloc(num_rstride(N, B, stride), sizeof(double));
  bc1d_copy_blockstride(X, Xs, N, B, stride);

  for(i = 0; i < N/B*stride; i++) {
    printf("%f\n", Xs[i]);
  }

  stride = getpagesize() / sizeof(double);

  free(X2r2);

  m2an = sizeof(double) * num_rstride(Nr, Br, stride) * Nc;

  X2r2 = (double *)malloc(m2an);

  bc2d_copy_blockstride(X2r, X2r2, Nr, Nc, Br, Bc, stride);

  fd1 = open("testfile2.dat", O_RDWR|O_CREAT, S_IREAD | S_IWRITE);
  write(fd1, X2r2, sizeof(double));
  
  ftruncate(fd1, m2an);

  X2m =  (double *)mmap(NULL, m2an, PROT_READ | PROT_WRITE, MAP_SHARED, fd1, 0);
  if((long)X2m == -1) { printf("Here1"); perror(NULL); exit(-226); }

  bc2d_copy_blockstride(X2r, X2m, Nr, Nc, Br, Bc, stride);

  msync(X2m, m2an, MS_SYNC);
  munmap(X2m, m2an);
  if(close(fd1) != 0) {
    printf("Here3"); 
    perror(NULL);
    exit(-223);
  }



  printf("\n\n");

  fd1 = open("testfile2.dat", O_RDONLY);
  
  X2m = (double *)mmap(NULL, m2an, PROT_READ, MAP_PRIVATE, fd1, 0);
  if((long)X2m == -1) { printf("Here2"); perror(NULL); exit(-226); }

  bc2d_copy_forward_stride(X2m, xc2, Nr, Nc, Br, Bc, Pr, Pc, pr, pc, stride);

  for(i = 0; i < nc; i++) {
    for(j = 0; j < nr; j++) {
      //printf("%10i %10.2f %10.2f\n", i, xc[i], X[i % B + B * (p + P * (i / B))]);
      printf("%5.1f  ", xc2[i*nr+j]);
    }
    printf("\n");
  }

  munmap(X2m, m2an);
  close(fd1);
  

  
  printf("\n\n");
  

}
