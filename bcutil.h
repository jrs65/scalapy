#ifndef _BCUTIL_H
#define _BCUTIL_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>



#define n_cblocks(N, B) ((N) / (B))

#define n_blocks(N, B) (ceildiv(N, B))

#define pid_remap(p, p0, P) (((p) + (P) - (p0)) % (P))

#define n_loc_cblocks(N, B, p, P) (n_cblocks(N, B) / (P) + n_cblocks(N, B) % (P) / ((p) + 1))


size_t num_c_blocks(size_t N, size_t B);
size_t num_blocks(size_t N, size_t B);

size_t n_c_lblocks(size_t N, size_t B, size_t p, size_t P);
size_t n_lblocks(size_t N, size_t B, size_t p, size_t P);

int partial_last_block(size_t N, size_t B, size_t p, size_t P);

size_t numrc(size_t N, size_t B, size_t p, size_t p0, size_t P);
size_t num_rstride(size_t N, size_t B, size_t stride);

size_t stride_page(size_t B);
size_t num_rpage(size_t N, size_t B);

int bc1d_copy_forward(double * src, double *dest, size_t N, size_t B, size_t P, size_t p);
int bc2d_copy_forward(double * src, double * dest, size_t Nr, size_t Nc, size_t Br, size_t Bc, size_t Pr, size_t Pc, size_t pr, size_t pc);


int bc1d_copy_backward(double * src, double *dest, size_t N, size_t B, size_t P, size_t p);
int bc2d_copy_backward(double * src, double * dest, size_t Nr, size_t Nc, size_t Br, size_t Bc, size_t Pr, size_t Pc, size_t pr, size_t pc);

int bc1d_copy_forward_stride(double * src, double *dest, size_t N, size_t B, size_t P, size_t p, size_t stride);
int bc1d_copy_backward_stride(double * src, double *dest, size_t N, size_t B, size_t P, size_t p, size_t stride);

int bc1d_copy_blockstride(double * src, double *dest, size_t N, size_t B, size_t stride);
int bc2d_copy_blockstride(double * src, double *dest, size_t Nr, size_t Nc, size_t Br, size_t Bc, size_t stride);

int bc1d_copy_pagealign(double * src, double * dest, size_t N, size_t B);
int bc2d_copy_pagealign(double * src, double * dest, size_t Nr, size_t Nc, size_t Br, size_t Bc);

int bc1d_from_pagealign(double * src, double *dest, size_t N, size_t B);
int bc2d_from_pagealign(double * src, double * dest, size_t Nr, size_t Nc, size_t Br, size_t Bc);

int bc2d_copy_forward_stride(double * src, double * dest, size_t Nr, size_t Nc, size_t Br, size_t Bc, size_t Pr, size_t Pc, size_t pr, size_t pc, size_t stride);
int bc2d_copy_backward_stride(double * src, double * dest, size_t Nr, size_t Nc, size_t Br, size_t Bc, size_t Pr, size_t Pc, size_t pr, size_t pc, size_t stride);

int bc1d_mmap_load(char * file, double * dest, size_t N, size_t B, size_t P, size_t p);
int bc2d_mmap_load(char * file, double * dest, size_t Nr, size_t Nc, size_t Br, size_t Bc, size_t Pr, size_t Pc, size_t pr, size_t pc);

int bc1d_mmap_save(char * file, double * src, size_t N, size_t B, size_t P, size_t p);
int bc2d_mmap_save(char * file, double * src, size_t Nr, size_t Nc, size_t Br, size_t Bc, size_t Pr, size_t Pc, size_t pr, size_t pc);

#endif
