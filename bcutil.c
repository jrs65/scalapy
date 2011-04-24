#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <fcntl.h>
#include <unistd.h>

#include <sys/mman.h>
#include <sys/stat.h>

#include <errno.h>



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
  return (nbc / P) + (((nbc % P) > p) ? 1 : 0);
}


int num_lblocks(int N, int B, int p, int P) {
  int nb = num_blocks(N, B);
  return (nb / P) + (((nb % P) > p) ? 1 : 0);
}

int partial_last_block(int N, int B, int p, int P) {
  return ((N % B > 0) && ((num_c_blocks(N, B) % P) == p));
}

int num_rstride(int N, int B, int stride) {
  return num_blocks(N, B) * stride;
}

int stride_page(int N, int B) {

  int pl, stride;

  pl = getpagesize() / sizeof(double);
  return ceildiv(B, pl) * pl;
}


int num_rpage(int N, int B) {
  return num_rstride(N, B, stride_page(N, B));
}


int numrc(int N, int B, int p, int p0, int P) {

  int nbp, n;

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
  return bc1d_copy_forward_stride(src, dest, N, B, P, p, B);
}

int bc2d_copy_forward(double * src, double * dest, int Nr, int Nc, int Br, int Bc, int Pr, int Pc, int pr, int pc) {
  return bc2d_copy_forward_stride(src, dest, Nr, Nc, Br, Bc, Pr, Pc, pr, pc, 0);
}



int bc1d_copy_backward(double * src, double *dest, int N, int B, int P, int p) {
  return bc1d_copy_backward_stride(src, dest, N, B, P, p, B);
}


int bc2d_copy_backward(double * src, double * dest, int Nr, int Nc, int Br, int Bc, int Pr, int Pc, int pr, int pc) {
  return bc2d_copy_backward_stride(src, dest, Nr, Nc, Br, Bc, Pr, Pc, pr, pc, 0);
}



int bc1d_copy_forward_stride(double * src, double *dest, int N, int B, int P, int p, int stride) {

  int b = 0;
  int lB;

  lB = num_c_lblocks(N, B, p, P); // Number of local, complete, blocks
  
  /* Iterate over complete local blocks, copying each from the full array into
     the sequential region in the local array */
  for(b = 0; b < lB; b++) {
    memcpy(dest + b*B, src + stride*(p + b*P), B*sizeof(double));
  }

  /* If there is a partial final block copy this */
  if(partial_last_block(N, B, p, P)) {
    memcpy(dest + lB*B,  src + stride*(p + lB*P), (N%B) * sizeof(double));
  }

  return 0;
}


int bc2d_copy_forward_stride(double * src, double * dest, int Nr, int Nc, int Br, int Bc, 
			     int Pr, int Pc, int pr, int pc, int stride) {

  int bc, i;
  int lBc;
  int nr;

  int ncs;

  lBc = num_c_lblocks(Nc, Bc, pc, Pc); // Number of local, complete, column blocks

  nr = numrc(Nr, Br, pr, 0, Pr); // Number of rows in the local array.
  
  if(stride == 0) {
    stride = Br;
    ncs = Nr;
  }
  else {
    ncs = num_rstride(Nr, Br, stride);
  }

  /* For each block, perform a 1d opy on its rows. */
  for(bc = 0; bc < lBc; bc++) {
    for(i = 0; i < Bc; i++) {
      bc1d_copy_forward_stride(src + (i + Bc*(pc + bc*Pc))*ncs,
			       dest + (bc*Bc + i)*nr, 
			       Nr, Br, Pr, pr, stride);
    }
  }

  /* Treat the special case of the final block */
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

  int b = 0;
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



int bc2d_copy_backward_stride(double * src, double * dest, int Nr, int Nc, int Br, int Bc,
			      int Pr, int Pc, int pr, int pc, int stride) {

  int bc, i;
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

  int b = 0;
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


int bc1d_copy_pagealign(double * src, double * dest, int N, int B) {
  int pl, stride;

  pl = getpagesize() / sizeof(double);
  stride = ceildiv(B, pl) * pl;

  return bc1d_copy_blockstride(src, dest, N, B, stride);
}


int bc2d_copy_pagealign(double * src, double * dest, int Nr, int Nc, int Br, int Bc) {
  int pl, stride;

  pl = getpagesize() / sizeof(double);
  stride = ceildiv(Bc, pl) * pl;

  return bc2d_copy_blockstride(src, dest, Nr, Nc, Br, Bc, stride);
}


int bc1d_mmap_load(char * file, double * dest, int N, int B, int P, int p) {

  int fd;

  double * xm;

  int pl, stride;
  long nm;

  struct stat fst;
  long fs;

  pl = getpagesize() / sizeof(double);
  stride = ceildiv(B, pl) * pl;
  nm = num_rstride(N, B, stride) * sizeof(double);

  fd = open(file, O_RDONLY);
  if(fd == -1) {
    perror("BC1d mmap load");
    return -1;
  }

  fstat(fd, &fst);
  fs = fst.st_size;
  if(fs < nm) {
    printf("File is not long enough.");
  }
  
  xm = (double *)mmap(NULL, nm, PROT_READ, MAP_PRIVATE, fd, 0);
  if((long)xm == -1) {
    perror("BC1d mmap load");
    return -1;
  }
  
  bc1d_copy_forward_stride(xm, dest, N, B, P, p, stride);

  munmap(xm, nm);
  close(fd);

  return 0;

}

int bc2d_mmap_load(char * file, double * dest, int Nr, int Nc, int Br, int Bc, int Pr, int Pc, int pr, int pc) {

  int fd;

  double * xm;

  int pl, stride; 
  long nm;

  struct stat fst;
  long fs;

  pl = getpagesize() / sizeof(double);
  stride = ceildiv(Br, pl) * pl;
  nm = num_rstride(Nr, Br, stride) * Nc * sizeof(double);

  printf("%i %i %i %i %i\n", stride, nm, pl, getpagesize(), sizeof(double));

  fd = open(file, O_RDONLY);
  if(fd == -1) {
    perror("BC2d mmap load");
    return -1;
  }

  fstat(fd, &fst);
  fs = fst.st_size;
  if(fs < nm) {
    printf("File is not long enough.");
  }
  
  xm = (double *)mmap(NULL, nm, PROT_READ, MAP_PRIVATE, fd, 0);
  if((long)xm == -1) {
    perror("BC2d mmap load");
    return -1;
  }
  
  bc2d_copy_forward_stride(xm, dest, Nr, Nc, Br, Bc, Pr, Pc, pr, pc, stride);

  munmap(xm, nm);
  close(fd);

  return 0;

}

int bc1d_mmap_save(char * file, double * src, int N, int B, int P, int p) {

  int fd;

  double * xm;

  int pl, stride;
  long nm;

  struct stat fst;
  long fs;

  pl = getpagesize() / sizeof(double);
  stride = ceildiv(B, pl) * pl;
  nm = num_rstride(N, B, stride) * sizeof(double);

  fd = open(file, O_RDWR);
  if(fd == -1) {
    perror("BC1d mmap save");
    return -1;
  }

  fstat(fd, &fst);
  fs = fst.st_size;
  if(fs < nm) {
    printf("File is not long enough.");
  }
  
  xm = (double *)mmap(NULL, nm, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if((long)xm == -1) {
    perror("BC1d mmap save");
    return -1;
  }
  
  bc1d_copy_backward_stride(src, xm, N, B, P, p, stride);

  msync(xm, nm, MS_SYNC);
  munmap(xm, nm);
  close(fd);

  return 0;

}



int bc2d_mmap_save(char * file, double * src, int Nr, int Nc, int Br, int Bc, int Pr, int Pc, int pr, int pc) {

  int fd;

  double * xm;

  int pl, stride; 
  long nm;

  struct stat fst;
  long fs;

  pl = getpagesize() / sizeof(double);
  stride = ceildiv(Br, pl) * pl;
  nm = num_rstride(Nr, Br, stride) * Nc * sizeof(double);


  fd = open(file, O_RDWR);
  if(fd == -1) {
    perror("BC2d mmap save");
    return -1;
  }

  fstat(fd, &fst);
  fs = fst.st_size;
  if(fs < nm) {
    printf("File is not long enough.");
  }
  
  xm = (double *)mmap(NULL, nm, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if((long)xm == -1) {
    perror("BC2d mmap save");
    return -1;
  }
  
  bc2d_copy_backward_stride(src, xm, Nr, Nc, Br, Bc, Pr, Pc, pr, pc, stride);

  msync(xm, nm, MS_SYNC);
  munmap(xm, nm);
  close(fd);

  return 0;

}
