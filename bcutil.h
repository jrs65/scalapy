#ifndef _BCUTIL_H
#define _BCUTIL_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>



#define n_cblocks(N, B) ((N) / (B))

#define n_blocks(N, B) (ceildiv(N, B))

#define pid_remap(p, p0, P) (((p) + (P) - (p0)) % (P))

#define n_loc_cblocks(N, B, p, P) (n_cblocks(N, B) / (P) + n_cblocks(N, B) % (P) / ((p) + 1))


int num_c_blocks(int N, int B);
int num_blocks(int N, int B);

int n_c_lblocks(int N, int B, int p, int P);
int n_lblocks(int N, int B, int p, int P);

int partial_last_block(int N, int B, int p, int P);

int numrc(int N, int B, int p, int p0, int P);
int num_rstride(int N, int B, int stride);

int stride_page(int N, int B);
int num_rpage(int N, int B);

int bc1d_copy_forward(double * src, double *dest, int N, int B, int P, int p);
int bc2d_copy_forward(double * src, double * dest, int Nr, int Nc, int Br, int Bc, int Pr, int Pc, int pr, int pc);


int bc1d_copy_backward(double * src, double *dest, int N, int B, int P, int p);
int bc2d_copy_backward(double * src, double * dest, int Nr, int Nc, int Br, int Bc, int Pr, int Pc, int pr, int pc);

int bc1d_copy_forward_stride(double * src, double *dest, int N, int B, int P, int p, int stride);
int bc1d_copy_backward_stride(double * src, double *dest, int N, int B, int P, int p, int stride);

int bc1d_copy_blockstride(double * src, double *dest, int N, int B, int stride);
int bc2d_copy_blockstride(double * src, double *dest, int Nr, int Nc, int Br, int Bc, int stride);

int bc1d_copy_pagealign(double * src, double * dest, int N, int B);
int bc2d_copy_pagealign(double * src, double * dest, int Nr, int Nc, int Br, int Bc);

int bc2d_copy_forward_stride(double * src, double * dest, int Nr, int Nc, int Br, int Bc, int Pr, int Pc, int pr, int pc, int stride);
int bc2d_copy_backward_stride(double * src, double * dest, int Nr, int Nc, int Br, int Bc, int Pr, int Pc, int pr, int pc, int stride);

int bc1d_mmap_load(char * file, double * dest, int N, int B, int P, int p);
int bc2d_mmap_load(char * file, double * dest, int Nr, int Nc, int Br, int Bc, int Pr, int Pc, int pr, int pc);

int bc1d_mmap_save(char * file, double * src, int N, int B, int P, int p);
int bc2d_mmap_save(char * file, double * src, int Nr, int Nc, int Br, int Bc, int Pr, int Pc, int pr, int pc);

#endif
