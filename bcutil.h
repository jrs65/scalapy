#ifndef _BCUTIL_H
#define _BCUTIL_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>


int numrc(int N, int B, int p, int p0, int P);




int bc1d_copy_forward(double * src, double *dest, int N, int B, int P, int p);

int bc2d_copy_forward(double * src, double * dest, int Nr, int Nc, int Br, int Bc, int Pr, int Pc, int pr, int pc);




int bc1d_copy_backward(double * src, double *dest, int N, int B, int P, int p);

int bc2d_copy_backward(double * src, double * dest, int Nr, int Nc, int Br, int Bc, int Pr, int Pc, int pr, int pc);


#endif
