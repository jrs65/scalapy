#include <stdio.h>
#include <string.h>
#include <stdlib.h>




#define ceildiv(x, y) ((x - 1) / (y) + 1)
#define pid_remap(p, p0, P) (((p) + (P) - (p0)) % (P))

int num_c_blocks(int N, int B) {
  return N / B;
}

int num_blocks(int N, int B) {
  return ceildiv(N, B);
}

int num_c_lblocks(int N, int B, int p, int P) {
  int nbc = num_c_blocks(N, B);
  return (nbc / P) + ((nbc % P) / (p + 1));
}


int num_lblocks(int N, int B, int p, int P) {
  int nb = num_blocks(N, B);
  return (nb / P) + ((nb % P) / (p + 1));
}

int partial_last_block(int N, int B, int p, int P) {
  return ((N % B > 0) && ((num_c_blocks(N, B) % P) == p));
}

int num_rstride(int N, int B, int stride) {
  return num_blocks(N, B) * stride;
}


int numrc(int N, int B, int p, int p0, int P) {

  int nbc, nbp, n;

  /* If the process owning block zero (p0) is not zero, then remap
     process numbers. */
  p = pid_remap(p, p0, P);
  
  /* Number of complete blocks owned by the process. */
  //nbp = (nbc - p - 1) / P + 1;
  nbp = num_c_lblocks(N, B, p, P);

  /* Number of entries of complete blocks owned by process. */
  n = nbp * B;
  
  /* If this process owns an incomplete block, then add the number of
     entries. */
  if(partial_last_block(N, B, p, P)) {
    n += N%B;
  }

  return n;

}


int bc1d_copy_forward(double * src, double *dest, int N, int B, int P, int p) {

  int b = 0, i;
  int lB;

  lB = num_c_lblocks(N, B, p, P); // Number of local, complete, blocks
  
  for(b = 0; b < lB; b++) {
    memcpy(dest + b*B, src + B*(p + b*P), B*sizeof(double));
    /*for(i = 0; i < B; i++) {
      dest[b*B+i] = src[B*(p+b*P)+i];
      printf("%i  %i  %f\n", p, b*B+i, dest[b*B+i]);
      }*/
  }

  if(partial_last_block(N, B, p, P)) {
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

  lBc = num_c_lblocks(Nc, Bc, pc, Pc); // Number of local, complete, col blocks

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

  if(partial_last_block(Nc, Bc, pc, Pc)) {
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

  lB = num_c_lblocks(N, B, p, P); // Number of local, complete, blocks
  
  for(b = 0; b < lB; b++) {
    memcpy(dest + B*(p + b*P), src + b*B, B*sizeof(double));
  }

  if(partial_last_block(N, B, p, P)) {
    memcpy(dest + N - N%B, src + lB*B, (N%B) * sizeof(double));
  }

  return 0;
}


int bc2d_copy_backward(double * src, double * dest, int Nr, int Nc, int Br, int Bc, int Pr, int Pc, int pr, int pc) {

  int bc, i, j;
  int lBc;
  int nr;

  lBc = num_c_lblocks(Nc, Bc, pc, Pc); // Number of local, complete, col blocks

  nr = numrc(Nr, Br, pr, 0, Pr);

  for(bc = 0; bc < lBc; bc++) {
    for(i = 0; i < Bc; i++) {
      bc1d_copy_backward(src + (bc*Bc + i)*nr, dest + (i + Bc*(pc + bc*Pc))*Nr, Nr, Br, Pr, pr);
    }
  }

  if(partial_last_block(Nc, Bc, pc, Pc)) {
    for(i = 0; i < Nc%Bc; i++) {
      bc1d_copy_backward(src + (lBc*Bc + i)*nr, dest + (i + Bc*(pc + lBc*Pc))*Nr, Nr, Br, Pr, pr);
    }
  }

  return 0;
}



int bc1d_copy_forward_stride(double * src, double *dest, int N, int B, int P, int p, int stride) {

  int b = 0, i;
  int lB;

  lB = num_c_lblocks(N, B, p, P); // Number of local, complete, blocks
  
  for(b = 0; b < lB; b++) {
    memcpy(dest + b*B, src + stride*(p + b*P), B*sizeof(double));
  }

  if(partial_last_block(N, B, p, P)) {
    memcpy(dest + lB*B,  src + stride*(p + lB*P), (N%B) * sizeof(double));
  }

  return 0;
}


int bc2d_copy_forward_stride(double * src, double * dest, int Nr, int Nc, int Br, int Bc, int Pr, int Pc, int pr, int pc, int stride) {

  int bc, i, j;
  int lBc;
  int nr;

  int ncs;

  lBc = num_c_lblocks(Nc, Bc, pc, Pc); // Number of local, complete, col blocks

  nr = numrc(Nr, Br, pr, 0, Pr);
  
  if(stride == 0) {
    stride = Br;
    ncs = Nr;
  }
  else {
    ncs = num_rstride(Nr, Br, stride);
  }

  for(bc = 0; bc < lBc; bc++) {
    for(i = 0; i < Bc; i++) {
      bc1d_copy_forward_stride(src + (i + Bc*(pc + bc*Pc))*ncs,
			       dest + (bc*Bc + i)*nr, 
			       Nr, Br, Pr, pr, stride);
    }
  }

  if(partial_last_block(Nc, Bc, pc, Pc)) {
    for(i = 0; i < Nc%Bc; i++) {
      bc1d_copy_forward_stride(src + (i + Bc*(pc + lBc*Pc))*ncs, 
			       dest + (lBc*Bc + i)*nr, 
			       Nr, Br, Pr, pr, stride);
    }
  }

  return 0;
}

int bc1d_copy_backward_stride(double * src, double *dest, int N, int B, int P, int p, int stride) {

  int b = 0, i;
  int lB;

  lB = num_c_lblocks(N, B, p, P); // Number of local, complete, blocks
  
  for(b = 0; b < lB; b++) {
    memcpy(dest + stride*(p + b*P), src + b*B, B*sizeof(double));
  }

  if(partial_last_block(N, B, p, P)) {
    memcpy(dest + stride*(p + lB*P), src + lB*B, (N%B) * sizeof(double));
  }

  return 0;
}



int bc2d_copy_backward_stride(double * src, double * dest, int Nr, int Nc, int Br, int Bc, int Pr, int Pc, int pr, int pc, int stride) {

  int bc, i, j;
  int lBc;
  int nr;

  int ncs;

  lBc = num_c_lblocks(Nc, Bc, pc, Pc); // Number of local, complete, col blocks

  nr = numrc(Nr, Br, pr, 0, Pr);

  if(stride == 0) {
    stride = Br;
    ncs = Nr;
  }
  else {
    ncs = num_rstride(Nr, Br, stride);
  }

  for(bc = 0; bc < lBc; bc++) {
    for(i = 0; i < Bc; i++) {
      bc1d_copy_backward_stride(src + (bc*Bc + i)*nr, dest + (i + Bc*(pc + bc*Pc))*ncs, Nr, Br, Pr, pr, stride);
    }
  }

  if(partial_last_block(Nc, Bc, pc, Pc)) {
    for(i = 0; i < Nc%Bc; i++) {
      bc1d_copy_backward_stride(src + (lBc*Bc + i)*nr, dest + (i + Bc*(pc + lBc*Pc))*ncs, Nr, Br, Pr, pr, stride);
    }
  }

  return 0;
}


int bc1d_copy_blockstride(double * src, double *dest, int N, int B, int stride) {

  int b = 0, i;
  int nB;

  nB = num_c_blocks(N, B); // Number of local, complete, blocks
  
  for(b = 0; b < nB; b++) {
    memcpy(dest + b*stride, src + b*B, B*sizeof(double));
  }

  if(N % B > 0) {
    memcpy(dest + nB*stride,  src + nB*B, (N%B) * sizeof(double));
  }

  return 0;
}


int bc2d_copy_blockstride(double * src, double * dest, int Nr, int Nc, int Br, int Bc, int stride) {

  int i;
  int ncs;

  ncs = num_rstride(Nr, Br, stride);

  for(i = 0; i < Nc; i++) {
    bc1d_copy_blockstride(src + i*Nr, dest + i*ncs, Nr, Br, stride);
  }

  return 0;
}
