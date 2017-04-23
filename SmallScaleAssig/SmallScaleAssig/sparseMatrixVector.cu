#include "SparseMatrixVector.cuh"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

template <unsigned int blockSize>
__device__ void warpReduce(volatile double *sdata, unsigned int tid) {
	if (blockSize >= 64)
		sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32)
		sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16)
		sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8)
		sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4)
		sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2)
		sdata[tid] += sdata[tid + 1];
}


template <unsigned int blockSize>
__global__ void reduceCSR(const int *JA, const double *AS, const double *x, int *IRP, double *output)
{
	unsigned int blockStart = IRP[blockIdx.x];
	unsigned int blockEnd = IRP[blockIdx.x + 1];
	unsigned int tid = threadIdx.x;
	__shared__ double values_in_row[blockSize];
	values_in_row[tid] = 0.0;

	for (int i = tid + blockStart; i < blockEnd; i += blockSize)
		values_in_row[tid] += AS[i] * x[JA[i]];

	__syncthreads();



	if (blockSize >= 512) {
		if (tid < 256) {
			values_in_row[tid] += values_in_row[tid + 256];
		}
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128) {
			values_in_row[tid] += values_in_row[tid + 128];
		}
		__syncthreads();
	}
	if (blockSize >= 128) {
		if (tid < 64) {
			values_in_row[tid] += values_in_row[tid + 64];
		}
		__syncthreads();
	}

	if (tid < 32)
		warpReduce<blockSize>(values_in_row, tid);

	// write result for this block to global mem
	if (tid == 0) {
		output[blockIdx.x] = values_in_row[0];
	}
}

template <unsigned int blockSize>
__global__ void reduceELL(const int *JA, const double *AS, const double *x, unsigned int maxNZ, double *output) {

	unsigned int tid = threadIdx.x;
	__shared__ double values_in_row[blockSize];

	unsigned int end = maxNZ * blockIdx.x + maxNZ;
	unsigned int index;
	values_in_row[tid] = 0.0;

	for (index = blockIdx.x * maxNZ + tid; index < end; index += blockSize)
		values_in_row[tid] += AS[index] * x[JA[index]];

	__syncthreads();

	if (blockSize >= 512) {
		if (tid < 256) {
			values_in_row[tid] += values_in_row[tid + 256];
		}
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128) {
			values_in_row[tid] += values_in_row[tid + 128];
		}
		__syncthreads();
	}
	if (blockSize >= 128) {
		if (tid < 64) {
			values_in_row[tid] += values_in_row[tid + 64];
		}
		__syncthreads();
	}

	if (tid < 32)
		warpReduce<blockSize>(values_in_row, tid);

	// write result for this block to global mem
	if (tid == 0) {
		output[blockIdx.x] = values_in_row[0];
	}
}

double * matrixVectorCSRCUDA(const int *IRP, const int *JA, const double *AS, const double *x, int M, const int N, const int nz, const int numRuns, double *average)
{
	//Host memory init 937807294
	double *result = (double *)malloc(M * sizeof(double));

	//Device memory init
	double *d_AS, *d_x, *d_y;
	int  *d_IRP, *d_JA;
	cudaMalloc((void**)&d_IRP, (M + 1) * sizeof(int));
	cudaMalloc((void**)&d_JA, nz * sizeof(int));
	cudaMalloc((void**)&d_AS, nz * sizeof(double));
	cudaMalloc((void**)&d_x, N * sizeof(double));
	cudaMalloc((void**)&d_y, M * sizeof(double));

	cudaMemcpy(d_IRP, IRP, (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_JA, JA, nz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_AS, AS, nz * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);

	const int THREADS_PER_BLOCK = 32;

	float time;
	double myAverage = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	for (int i = 0; i < numRuns; i++) {
		cudaEventRecord(start, 0);
		reduceCSR<THREADS_PER_BLOCK> << <M, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double) >> > (d_JA, d_AS, d_x, d_IRP, d_y);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		myAverage += time;
	}
	myAverage /= (double)numRuns;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	*average = myAverage;

	cudaMemcpy(result, d_y, M * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_IRP);
	cudaFree(d_JA);
	cudaFree(d_AS);
	cudaFree(d_x);
	cudaFree(d_y);

	return result;
}

double * matrixVectorELLCUDA(const int maxNZ, const int *JA, const double *AS, const double *x, int M, const int N, const int nz, const int numRuns, double *average)
{
	//Host memory init 937807294
	double *result = (double *)malloc(M * sizeof(double));

	//Device memory init
	double *d_AS, *d_x, *d_y;
	int *d_JA;
	cudaMalloc((void**)&d_JA, M * maxNZ * sizeof(int));
	cudaMalloc((void**)&d_AS, M * maxNZ * sizeof(double));
	cudaMalloc((void**)&d_x, N * sizeof(double));
	cudaMalloc((void**)&d_y, M * sizeof(double));

	cudaMemcpy(d_JA, JA, M * maxNZ * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_AS, AS, M * maxNZ * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);

	const unsigned int THREADS_PER_BLOCK = 32;

	float time;
	double myAverage = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	for (int i = 0; i < numRuns; i++) {
		cudaEventRecord(start, 0);
		reduceELL<THREADS_PER_BLOCK> << <M, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double) >> > (d_JA, d_AS, d_x, maxNZ, d_y);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		myAverage += time;
	}
	myAverage /= (double)numRuns;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	*average = myAverage;

	cudaMemcpy(result, d_y, M * sizeof(double), cudaMemcpyDeviceToHost);
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.


	cudaFree(d_JA);
	cudaFree(d_AS);
	cudaFree(d_x);
	cudaFree(d_y);

	return result;
}