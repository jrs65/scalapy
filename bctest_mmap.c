#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <fcntl.h>
#include <unistd.h>

#include <sys/mman.h>

#include <errno.h>

#include "bcutil.h"



int main(int argc, char **argv) {


  double * Xr, * Xc, * Xp;
  double * Xrc;

  double * xr;

  int i, j, nr, nc;

  int Nc = 17, Nr = 11, Bc = 3, Br = 2, Pc = 3, Pr = 3, pc = 2, pr = 2;

  int pi, pj, nm, fd;

  printf("\n2d test\n");


  Xr = (double *)malloc(sizeof(double) * Nr * Nc);
  Xrc = (double *)malloc(sizeof(double) * Nr * Nc);

  for(i = 0; i < Nc; i++) {
    for(j = 0; j < Nr; j++) {
      //Xc[i*Nr + j] = i;
      Xr[i*Nr + j] = j;
    }
  }
  
  nm = sizeof(double) * num_rpage(Nr, Br) * Nc;

  Xp = (double *)malloc(nm);

  bc2d_copy_pagealign(Xr, Xp, Nr, Nc, Br, Bc);

  fd = open("testmm1.dat", O_WRONLY|O_CREAT, S_IREAD | S_IWRITE);
  if(write(fd, Xp, nm) == -1) {
    perror(NULL);
    exit(-30);
  }
  close(fd);
  
  fd = open("testmm2.dat", O_WRONLY|O_CREAT, S_IREAD | S_IWRITE);
  ftruncate(fd, nm);
  close(fd);

  // MMAP Read in.
  for(pi = 0; pi < Pc; pi++) {
    for(pj = 0; pj < Pr; pj++) {
      nc = numrc(Nc, Bc, pi, 0, Pc);
      nr = numrc(Nr, Br, pj, 0, Pr);
      xr = (double *)malloc(sizeof(double) * nc * nr);

      bc2d_mmap_load("testmm1.dat", xr, Nr, Nc, Br, Bc, Pr, Pc, pj, pi);
      bc2d_copy_backward(xr, Xrc, Nr, Nc, Br, Bc, Pr, Pc, pj, pi);

      bc2d_mmap_save("testmm2.dat", xr, Nr, Nc, Br, Bc, Pr, Pc, pj, pi);

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



}
