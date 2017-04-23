#include "OpenMP.h"
#include <iostream>

void matrixVectorCSROpenMP(int *IRP, int *JA, double *AS, double *x, int M, int nthreads, double *result) {

	omp_set_num_threads(nthreads);
	int i, j;
	double temp;

#pragma omp parallel shared(result,i) private(j, temp)
	{
#pragma omp for schedule(static)
		for (i = 0; i < M; i++)
		{
			temp = 0.0;
			for (j = IRP[i]; j < IRP[i + 1]; j++)
			{
				temp += AS[j] * x[JA[j]];
			}
			result[i] = temp;
		}
	}
}

void matrixVectorELLOpenMP(int maxNZ, int * JA, double * AS, double * x, int M, int nthreads, double *result)
{
	omp_set_num_threads(nthreads);
	int i, j, index;
	double temp;
#pragma omp parallel shared(result,i) private(j, index, temp)
	{
#pragma omp for schedule(static)
		for (i = 0; i < M; i++)
		{
			temp = 0.0;
			index = i * maxNZ;
			for (j = 0; j < maxNZ; j++)
			{
				temp += AS[index] * x[JA[index]];
				index++;
			}
			result[i] = temp;
		}
	}
}
