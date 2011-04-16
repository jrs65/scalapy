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

  int Nc = 17, Nr = 5, Bc = 3, Br = 2, Pc = 3, Pr = 2, pc = 2, pr = 0;

  int pi, pj;

  int fd, fd1, fd2;

  double * X2m;

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

  fd = open("testfile1.dat", O_WRONLY|O_CREAT);
  if(write(fd, X2r, sizeof(double)*Nr*Nc) == -1) {
    printf("Error.\n");
    perror(NULL);
    exit(-30);
  }
  close(fd);

  fd1 = open("testfile1.dat", O_RDONLY);
  
  nr = numrc(Nr, Br, pr, 0, Pr);
  nc = numrc(Nc, Bc, pc, 0, Pc);

  X2m = (double *)mmap(NULL, sizeof(double) * Nr* Nc, PROT_READ, MAP_PRIVATE, fd1, 0);
  if((int)X2m == -1) { perror(NULL); exit(-226); }

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

}
