#include "cuda_runtime.h"


double * matrixVectorCSRCUDA(const int *IRP, const int *JA, const double *AS, const double *x, const int M, const int N, const int nz, const int numRuns, double *average);
double * matrixVectorELLCUDA(const int maxNZ, const int *JA, const double *AS, const double *x, int M, const int N, const int nz, const int numRuns, double *average);