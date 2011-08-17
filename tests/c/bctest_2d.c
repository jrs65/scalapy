#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <fcntl.h>
#include <unistd.h>

#include <sys/mman.h>

#include <errno.h>

#include "bcutil.h"



int main(int argc, char **argv) {

  double * Xr, * Xc;
  double * Xrc;

  double * xr, * xc;

  int i, j, nr, nc;

  int Nc = 17, Nr = 11, Bc = 3, Br = 2, Pc = 3, Pr = 3, pc = 2, pr = 2;

  int pi, pj;

  printf("\n2d test\n");


  Xr = (double *)malloc(sizeof(double) * Nr * Nc);
  Xrc = (double *)malloc(sizeof(double) * Nr * Nc);
  Xc = (double *)malloc(sizeof(double) * Nr * Nc);

  for(i = 0; i < Nc; i++) {
    for(j = 0; j < Nr; j++) {
      Xc[i*Nr + j] = i;
      Xr[i*Nr + j] = j;
    }
  }
  
  nr = numrc(Nr, Br, pr, 0, Pr);
  nc = numrc(Nc, Bc, pc, 0, Pc);
  xc = (double *)malloc(sizeof(double) * nr * nc);
  xr = (double *)malloc(sizeof(double) * nr * nc);

  printf("Local array size: %i x %i\n", nc, nr);

  bc2d_copy_forward(Xc, xc, Nr, Nc, Br, Bc, Pr, Pc, pr, pc);
  bc2d_copy_forward(Xr, xr, Nr, Nc, Br, Bc, Pr, Pc, pr, pc);

  for(i = 0; i < nc; i++) {
    for(j = 0; j < nr; j++) {
      printf("%4.1f ", xc[i*nr+j]);
    }
    printf("\n");
  }
  printf("\n\n");
  for(i = 0; i < nc; i++) {
    for(j = 0; j < nr; j++) {
      printf("%4.1f ", xr[i*nr+j]);
    }
    printf("\n");
  }
  printf("\n\n");
  free(xr);
  free(xc);

  for(pi = 0; pi < Pc; pi++) {
    for(pj = 0; pj < Pr; pj++) {
      nc = numrc(Nc, Bc, pi, 0, Pc);
      nr = numrc(Nr, Br, pj, 0, Pr);
      xr = (double *)malloc(sizeof(double) * nc * nr);

      bc2d_copy_forward(Xr, xr, Nr, Nc, Br, Bc, Pr, Pc, pj, pi);
      bc2d_copy_backward(xr, Xrc, Nr, Nc, Br, Bc, Pr, Pc, pj, pi);

      free(xr);
    }
  }

  for(i = 0; i < Nc; i++) {
    for(j = 0; j < Nr; j++) {
      printf("%4.1f ", Xr[i*Nr+j]);
    }
    printf("\n");
  }

  printf("\n\n");

  for(i = 0; i < Nc; i++) {
    for(j = 0; j < Nr; j++) {
      printf("%4.1f ", Xrc[i*Nr+j]);
    }
    printf("\n");
  }

  free(Xr);
  free(Xc);
  free(Xrc);
  
  return 0;
}
