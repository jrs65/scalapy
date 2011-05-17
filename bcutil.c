#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <fcntl.h>
#include <unistd.h>

#include <sys/mman.h>
#include <sys/stat.h>

#include <errno.h>

#include <math.h>


#define ceildiv(x, y) ((x - 1) / (y) + 1)
#define pid_remap(p, p0, P) (((p) + (P) - (p0)) % (P))

size_t num_c_blocks(size_t N, size_t B) {
  return N / B;
}

size_t num_blocks(size_t N, size_t B) {
  return ceildiv(N, B);
}

size_t num_c_lblocks(size_t N, size_t B, size_t p, size_t P) {
  size_t nbc = num_c_blocks(N, B);
  return (nbc / P) + (((nbc % P) > p) ? 1 : 0);
}


size_t num_lblocks(size_t N, size_t B, size_t p, size_t P) {
  size_t nb = num_blocks(N, B);
  return (nb / P) + (((nb % P) > p) ? 1 : 0);
}

int partial_last_block(size_t N, size_t B, size_t p, size_t P) {
  return ((N % B > 0) && ((num_c_blocks(N, B) % P) == p));
}

size_t num_rstride(size_t N, size_t B, size_t stride) {
  return num_blocks(N, B) * stride;
}

size_t stride_page(size_t B) {

  size_t pl;

  pl = (size_t) sysconf (_SC_PAGESIZE) / sizeof(double);
  return ceildiv(B, pl) * pl;
}


size_t num_rpage(size_t N, size_t B) {
  return num_rstride(N, B, stride_page(B));
}


size_t numrc(size_t N, size_t B, size_t p, size_t p0, size_t P) {

  size_t nbp, n;

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

int indices_rc(size_t N, size_t B, size_t p, size_t P, int * ind) {

  size_t nt, nb;
  int i, j;

  nt = numrc(N, B, p, 0, P);
  nb = num_c_lblocks(N, B, p, P);

  for(i = 0; i < nb; i++) {
    for(j = 0; j < B; j++) {
      ind[i*B+j] = (i*P+p)*B+j;
    }
  }
  
  if(nb * B < nt) {
    for(j = 0; j < B; j++) {
      ind[nb*B+j] = (nb*P+p)*B+j;
    }
  }
  
  return 0;
  
}

/*
int scinit(int argc, char ** argv, int * ictxt, int * Pr, int * Pc, int * pr, int * pc, int * rank, int * size) {
  MPI_Init(&argc, &argv);
  //MPI_Comm_size(MPI_COMM_WORLD, size);
  //MPI_Comm_rank(MPI_COMM_WORLD,rank);

  Cblacs_pinfo( rank, size );
  printf("RS: %i %i\n", *rank, *size);

  *Pr = (int)sqrt(*size);
  *Pc = (int)sqrt(*size);

  printf("Px: %i %i\n", *Pr, *Pc);

  Cblacs_get( -1, 0, ictxt );
  printf("Ic: %i\n", *ictxt);
  Cblacs_gridinit( ictxt, "Row", *Pr, *Pc );
  printf("hello\n");
  Cblacs_gridinfo( *ictxt, Pr, Pc, pr, pc );
  printf("hello2\n");
  return 0;
}

*/


int bc1d_copy_forward_stride(double * src, double *dest, size_t N, size_t B, size_t P, size_t p, size_t stride) {

  size_t b = 0;
  size_t lB;

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


int bc2d_copy_forward_stride(double * src, double * dest, size_t Nr, size_t Nc, size_t Br, size_t Bc, 
			     size_t Pr, size_t Pc, size_t pr, size_t pc, size_t stride) {

  size_t bc, i;
  size_t lBc;
  size_t nr;

  size_t ncs;

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

int bc1d_copy_backward_stride(double * src, double *dest, size_t N, size_t B, size_t P, size_t p, size_t stride) {

  size_t b = 0;
  size_t lB;

  lB = num_c_lblocks(N, B, p, P); // Number of local, complete, blocks
  
  for(b = 0; b < lB; b++) {
    memcpy(dest + stride*(p + b*P), src + b*B, B*sizeof(double));
  }

  if(partial_last_block(N, B, p, P)) {
    memcpy(dest + stride*(p + lB*P), src + lB*B, (N%B) * sizeof(double));
  }

  return 0;
}



int bc2d_copy_backward_stride(double * src, double * dest, size_t Nr, size_t Nc, size_t Br, size_t Bc,
			      size_t Pr, size_t Pc, size_t pr, size_t pc, size_t stride) {

  size_t bc, i;
  size_t lBc;
  size_t nr;

  size_t ncs;

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


int bc1d_copy_forward(double * src, double *dest, size_t N, size_t B, size_t P, size_t p) {
  return bc1d_copy_forward_stride(src, dest, N, B, P, p, B);
}


int bc2d_copy_forward(double * src, double * dest, size_t Nr, size_t Nc, size_t Br, size_t Bc, size_t Pr, size_t Pc, size_t pr, size_t pc) {
  return bc2d_copy_forward_stride(src, dest, Nr, Nc, Br, Bc, Pr, Pc, pr, pc, 0);
}



int bc1d_copy_backward(double * src, double *dest, size_t N, size_t B, size_t P, size_t p) {
  return bc1d_copy_backward_stride(src, dest, N, B, P, p, B);
}


int bc2d_copy_backward(double * src, double * dest, size_t Nr, size_t Nc, size_t Br, size_t Bc, size_t Pr, size_t Pc, size_t pr, size_t pc) {
  return bc2d_copy_backward_stride(src, dest, Nr, Nc, Br, Bc, Pr, Pc, pr, pc, 0);
}




int bc1d_copy_blockstride(double * src, double *dest, size_t N, size_t B, size_t stride) {

  return bc1d_copy_backward_stride(src, dest, N, B, 1, 0, stride);
}


int bc2d_copy_blockstride(double * src, double * dest, size_t Nr, size_t Nc, size_t Br, size_t Bc, size_t stride) {

  return bc2d_copy_backward_stride(src, dest, Nr, Nc, Br, Bc, 1, 1, 0, 0, stride);
}


int bc1d_copy_pagealign(double * src, double * dest, size_t N, size_t B) {

  return bc1d_copy_blockstride(src, dest, N, B, stride_page(B));
}


int bc2d_copy_pagealign(double * src, double * dest, size_t Nr, size_t Nc, size_t Br, size_t Bc) {

  return bc2d_copy_blockstride(src, dest, Nr, Nc, Br, Bc, stride_page(Br));
}


int bc1d_from_pagealign(double * src, double *dest, size_t N, size_t B) {

  return bc1d_copy_forward_stride(src, dest, N, B, 1, 0, stride_page(B));
}


int bc2d_from_pagealign(double * src, double * dest, size_t Nr, size_t Nc, size_t Br, size_t Bc) {

  return bc2d_copy_forward_stride(src, dest, Nr, Nc, Br, Bc, 1, 1, 0, 0, stride_page(Br));
}



int bc1d_mmap_load(char * file, double * dest, size_t N, size_t B, size_t P, size_t p) {

  int fd;

  double * xm;

  size_t stride;
  size_t nm;

  struct stat fst;
  size_t fs;

  stride = stride_page(B);
  nm = num_rpage(N, B) * sizeof(double);

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

int bc2d_mmap_load(char * file, double * dest, size_t Nr, size_t Nc, size_t Br, size_t Bc, size_t Pr, size_t Pc, size_t pr, size_t pc) {

  int fd;

  double * xm;

  size_t stride; 
  size_t nm;

  struct stat fst;
  size_t fs;

  stride = stride_page(Br);
  nm = num_rpage(Nr, Br) * Nc * sizeof(double);

  //printf("%i %i %i %i %i\n", stride, nm, pl, getpagesize(), sizeof(double));

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

int bc1d_mmap_save(char * file, double * src, size_t N, size_t B, size_t P, size_t p) {

  int fd;

  double * xm;

  size_t stride;
  size_t nm;

  struct stat fst;
  size_t fs;

  stride = stride_page(B);
  nm = num_rpage(N, B) * sizeof(double);

  fd = open(file, O_RDWR);
  if(fd == -1) {
    perror("BC1d mmap save");
    return -1;
  }

  fstat(fd, &fst);
  fs = fst.st_size;
  if(fs < nm) {
    printf("File is not long enough.");
    exit(-29);
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



int bc2d_mmap_save(char * file, double * src, size_t Nr, size_t Nc, size_t Br, size_t Bc, size_t Pr, size_t Pc, size_t pr, size_t pc) {

  int fd;

  double * xm;

  size_t stride; 
  size_t nm;

  struct stat fst;
  size_t fs;

  stride = stride_page(Br);
  nm = num_rpage(Nr, Br) * Nc * sizeof(double);


  fd = open(file, O_RDWR);
  if(fd == -1) {
    perror("BC2d mmap save");
    return -1;
  }

  fstat(fd, &fst);
  fs = fst.st_size;
  if(fs < nm) {
    printf("File is not long enough.");
    exit(-29);
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
