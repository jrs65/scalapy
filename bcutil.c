#include <stdio.h>
#include <string.h>
#include <stdlib.h>


int numrc(int N, int B, int p, int p0, int P) {

  int nbc, nbp, n;

  /* If the process owning block zero (p0) is not zero, then remap
     process numbers. */
  p = (p + p0) % P;

  /* Number of complete blocks. */
  nbc = N / B;
  
  /* Number of complete blocks owned by the process. */
  //nbp = (nbc - p - 1) / P + 1;
  nbp = nbc / P + (nbc % P) / (p+1);

  /* Number of entries of complete blocks owned by process. */
  n = nbp * B;
  
  /* If this process owns an incomplete block, then add the number of
     entries. */
  if(N % B > 0 && (nbc % P) == p) {
    n += N%B;
  }

  return n;

}


int bc1d_copy_forward(double * src, double *dest, int N, int B, int P, int p) {

  int b = 0, i;
  int lB;

  lB = ((N / B) - p - 1) / P + 1; // Number of local, complete, blocks
  
  for(b = 0; b < lB; b++) {
    memcpy(dest + b*B, src + B*(p + b*P), B*sizeof(double));
    /*for(i = 0; i < B; i++) {
      dest[b*B+i] = src[B*(p+b*P)+i];
      printf("%i  %i  %f\n", p, b*B+i, dest[b*B+i]);
      }*/
  }

  if(N % B > 0 && (N/B % P) == p) {
    //printf("Here %i\n", p);
    memcpy(dest + lB*B, src + N - N%B, (N%B) * sizeof(double));
    /*for(i = 0; i < N%B; i++) {
      dest[lB*B+i] = src[B*(p+lB*P)+i];
      printf("%i  %i  %f\n", p, lB*B+i, dest[lB*B+i]);
      }*/
  }

  return 0;
}

int bc2d_copy_forward(double * src, double * dest, int Nr, int Nc, int Br, int Bc, int Pr, int Pc, int pr, int pc) {

  int bc, i, j;
  int lBc;
  int nr;

  lBc = ((Nc / Bc) - pc - 1) / Pc + 1; // Number of local, complete, col blocks

  nr = numrc(Nr, Br, pr, 0, Pr);

  for(bc = 0; bc < lBc; bc++) {
    for(i = 0; i < Bc; i++) {
      /*printf("\nColumn %i %i\n", bc, i);
      for(j = 0; j < Nr; j++) {
        printf("%10.2f\n", src[(i + Bc*(pc + bc*Pc))*Nr+j]);
      }
      printf("\n");*/
      bc1d_copy_forward(src + (i + Bc*(pc + bc*Pc))*Nr, dest + (bc*Bc + i)*nr, Nr, Br, Pr, pr);
      /*for(j = 0; j < nr; j++) {
        printf("%10.2f\n", dest[(i + bc*Bc)*nr+j]);
        }*/
    }
  }

  if(Nc % Bc > 0 && (Nc/Bc % Pc) == pc) {
    for(i = 0; i < Nc%Bc; i++) {
      /* printf("\nColumn %i %i\n", lBc, i); */
      /* for(j = 0; j < Nr; j++) { */
      /*   printf("%10.2f\n", src[(i + Bc*(pc + lBc*Pc))*Nr+j]); */
      /* } */
      /* printf("\n"); */
      bc1d_copy_forward(src + (i + Bc*(pc + lBc*Pc))*Nr, dest + (lBc*Bc + i)*nr, Nr, Br, Pr, pr);
      /* for(j = 0; j < nr; j++) { */
      /*   printf("%10.2f\n", dest[(i + lBc*Bc)*nr+j]); */
      /* } */
    }
  }

  return 0;
}



int bc1d_copy_backward(double * src, double *dest, int N, int B, int P, int p) {

  int b = 0, i;
  int lB;

  lB = ((N / B) - p - 1) / P + 1; // Number of local, complete, blocks
  
  for(b = 0; b < lB; b++) {
    memcpy(dest + B*(p + b*P), src + b*B, B*sizeof(double));
  }

  if(N % B > 0 && (N/B % P) == p) {
    memcpy(dest + N - N%B, src + lB*B, (N%B) * sizeof(double));
  }

  return 0;
}


int bc2d_copy_backward(double * src, double * dest, int Nr, int Nc, int Br, int Bc, int Pr, int Pc, int pr, int pc) {

  int bc, i, j;
  int lBc;
  int nr;

  lBc = ((Nc / Bc) - pc - 1) / Pc + 1; // Number of local, complete, col blocks

  nr = numrc(Nr, Br, pr, 0, Pr);

  for(bc = 0; bc < lBc; bc++) {
    for(i = 0; i < Bc; i++) {
      bc1d_copy_backward(src + (bc*Bc + i)*nr, dest + (i + Bc*(pc + bc*Pc))*Nr, Nr, Br, Pr, pr);
    }
  }

  if(Nc % Bc > 0 && (Nc/Bc % Pc) == pc) {
    for(i = 0; i < Nc%Bc; i++) {
      bc1d_copy_backward(src + (lBc*Bc + i)*nr, dest + (i + Bc*(pc + lBc*Pc))*Nr, Nr, Br, Pr, pr);
    }
  }

  return 0;
}


