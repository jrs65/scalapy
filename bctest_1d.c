#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <fcntl.h>
#include <unistd.h>

#include <sys/mman.h>

#include <errno.h>

#include <stddef.h>

#include "bcutil.h"



int main(int argc, char **argv) {
  double * X, *Xc; 
  double * xc;

  long t1i, t1l;
  int t2i = 1000000000;
  long t2l = 1000000000;

  int i, nr, pi;

  int N = 19, B = 3, P = 3, p = 0;

  printf("\n1d test\n");

  X = (double *)malloc(sizeof(double) * N);
  Xc = (double *)malloc(sizeof(double) * N);

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

  free(xc);

  printf("\n\n");
  for(pi = 0; pi < P; pi++) {
    nr = numrc(N, B, pi, 0, P);
    xc = (double *)malloc(sizeof(double) * nr);
    
    bc1d_copy_forward(X, xc, N, B, P, pi);
    bc1d_copy_backward(xc, Xc, N, B, P, pi);
    
    free(xc);
  }

  
  for(i = 0; i < N; i++) {
    printf("%4.1f ", X[i]);
  }
  printf("\n\n");
  for(i = 0; i < N; i++) {
    printf("%4.1f ", Xc[i]);
  }
  printf("\n");

  free(X);
  free(Xc);
  X = NULL;
  Xc = NULL;

  t1i = t2i *t2i + t2l;
  t1l = t2l *t2l + t2l;

  printf("i %ld\nl %ld\n", t1i, t1l);
  printf("si %i\n", sizeof(int));
  printf("su %i\n", sizeof(unsigned int));
  printf("st %i\n", sizeof(size_t));
  printf("pd %i\n", sizeof(ptrdiff_t));
  printf("ds %i\n", sizeof(double *));
  return 0;

}
