
ctypedef unsigned long size_t

cdef extern from "bcutil.h":
    int bc1d_copy_pagealign(double * src, double * dest, int N, int B)
    int bc2d_copy_pagealign(double * src, double * dest, int Nr, int Nc, int Br, int Bc)
    int bc1d_from_pagealign(double * src, double *dest, size_t N, size_t B)
    int bc2d_from_pagealign(double * src, double * dest, size_t Nr, size_t Nc, size_t Br, size_t Bc)
    int num_rpage(int N, int B)

    size_t numrc(size_t N, size_t B, size_t p, size_t p0, size_t P)
    int bc1d_mmap_load(char * file, double * dest, size_t N, size_t B, size_t P, size_t p)
    int bc2d_mmap_load(char * file, double * dest, size_t Nr, size_t Nc, size_t Br, size_t Bc, size_t Pr, size_t Pc, size_t pr, size_t pc)

    int bc1d_mmap_save(char * file, double * src, size_t N, size_t B, size_t P, size_t p)
    int bc2d_mmap_save(char * file, double * src, size_t Nr, size_t Nc, size_t Br, size_t Bc, size_t Pr, size_t Pc, size_t pr, size_t pc)

    
    int bc1d_copy_forward(double * src, double *dest, size_t N, size_t B, size_t P, size_t p)
    int bc2d_copy_forward(double * src, double * dest, size_t Nr, size_t Nc, size_t Br, size_t Bc, size_t Pr, size_t Pc, size_t pr, size_t pc)


    int bc1d_copy_backward(double * src, double *dest, size_t N, size_t B, size_t P, size_t p)
    int bc2d_copy_backward(double * src, double * dest, size_t Nr, size_t Nc, size_t Br, size_t Bc, size_t Pr, size_t Pc, size_t pr, size_t pc)

    int scinit(int argc, char ** argv, int * ictxt, int * Pr, int * Pc, int * pr, int * pr, int * rank, int * size)

