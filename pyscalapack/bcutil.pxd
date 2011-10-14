
ctypedef unsigned long size_t

cdef extern from "bcutil.h":
    int bc1d_copy_pagealign(char * src, char * dest, size_t itemsize, int N, int B)
    int bc2d_copy_pagealign(char * src, char * dest, size_t itemsize, int Nr, int Nc, int Br, int Bc)
    int bc1d_from_pagealign(char * src, char *dest, size_t itemsize, size_t N, size_t B)
    int bc2d_from_pagealign(char * src, char * dest, size_t itemsize, size_t Nr, size_t Nc, size_t Br, size_t Bc)
    int num_rpage(int N, int B, size_t itemsize)

    int indices_rc(size_t N, size_t B, size_t p, size_t P, int * ind)
    
    size_t numrc(size_t N, size_t B, size_t p, size_t p0, size_t P)
    int bc1d_mmap_load(char * file, char * dest, size_t itemsize, size_t N, size_t B, size_t P, size_t p)
    int bc2d_mmap_load(char * file, char * dest, size_t itemsize, size_t Nr, size_t Nc, size_t Br, size_t Bc, size_t Pr, size_t Pc, size_t pr, size_t pc)

    int bc1d_mmap_save(char * file, char * src, size_t itemsize, size_t N, size_t B, size_t P, size_t p)
    int bc2d_mmap_save(char * file, char * src, size_t itemsize, size_t Nr, size_t Nc, size_t Br, size_t Bc, size_t Pr, size_t Pc, size_t pr, size_t pc)

    
    int bc1d_copy_forward(char * src, char *dest, size_t itemsize, size_t N, size_t B, size_t P, size_t p)
    int bc2d_copy_forward(char * src, char * dest, size_t itemsize, size_t Nr, size_t Nc, size_t Br, size_t Bc, size_t Pr, size_t Pc, size_t pr, size_t pc)


    int bc1d_copy_backward(char * src, char *dest, size_t itemsize, size_t N, size_t B, size_t P, size_t p)
    int bc2d_copy_backward(char * src, char * dest, size_t itemsize, size_t Nr, size_t Nc, size_t Br, size_t Bc, size_t Pr, size_t Pc, size_t pr, size_t pc)


