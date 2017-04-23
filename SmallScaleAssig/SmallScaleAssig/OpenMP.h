#pragma once
#include <omp.h>

void matrixVectorCSROpenMP(int *IRP, int *JA, double *AS, double *x, int M, int nthreads, double *result);

void matrixVectorELLOpenMP(int maxNZ, int *JA, double *AS, double *x, int M, int nthreads, double *result);