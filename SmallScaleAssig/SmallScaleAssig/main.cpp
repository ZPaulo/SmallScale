#include "Matrix.h"
#include <cmath>
#include "OpenMP.h"
#include <iostream>
#include "SparseMatrixVector.cuh"
#include <dirent.h>
#include <sys/time.h>
#include <stdio.h>
#include <string.h>
#include "unistd.h"

const int MAX_THREADS = 32;

using namespace std;

double serialMMCSR(const int *IRP, const int *JA, const double *AS, const double *x, const int M, double *result)
{
	int i, j;
	double temp;
	for (i = 0; i < M; i++)
	{
		temp = 0;
		for (j = IRP[i]; j < IRP[i + 1]; j++)
		{
			temp += AS[j] * x[JA[j]];

		}
		result[i] = temp;
	}
	return 0;
}

double serialMMELLPACK(const int maxNZ, const int *JA, const double *AS, const double *x, const int M, double *result)
{
	int i, j, index;
	double temp;

	for (i = 0; i < M; i++)
	{
		temp = 0;
		for (j = 0; j < maxNZ; j++)
		{
			index = i * maxNZ + j;
			temp += AS[index] * x[JA[index]];
		}
		result[i] = temp;
	}

	return 0;
}

void writeToFileCUDA(double serialCSR, double parallelCSR, double serialELL, double parallelELL, char const *matrix, int nz, double maxErrorCSR, double maxErrorEll) {
	FILE *file = fopen("CUDA.csv", "a");
	fprintf(file, "%s,%d, ,%f,%f, ,%f,%f, ,%f,%f\n", matrix, nz, serialCSR, parallelCSR, serialELL, parallelELL, maxErrorCSR, maxErrorEll);
	fclose(file);
}

void writeToFileOpenMP(double serialCSR, double *parallelCSR, double serialELL, double *parallelELL, char const *matrix, double n, int nz, double maxErrorCSR, double maxErrorEll) {
	FILE *file = fopen("OpenMP.csv", "a");
	fprintf(file, "%s, %d, ,%f,%f, ,%f,%f", matrix, nz, maxErrorCSR, maxErrorEll, serialCSR, serialELL);
	for (unsigned int i = 0; i < n; i++)
	{
		if (parallelELL != NULL)
			fprintf(file, ",%f,%f", parallelCSR[i], parallelELL[i]);
		else
			fprintf(file, ",%f,-1", parallelCSR[i]);
	}
	fprintf(file, "\n");
	fclose(file);
}

void createFileCuda() {

	if (access("CUDA.csv", F_OK) == -1) {
	FILE *file = fopen("CUDA.csv", "a");
	fprintf(file, "Matrix,NonZeros, ,CSR,, ,ELLPACK,, ,MaxErrorCSR,MaxErrorELL\n");
	fprintf(file, ",, ,Serial,Parallel, ,Serial,Parallel\n");
	fclose(file);
	}	
}

void createFileOpenMP() {
	if (access("OpenMP.csv", F_OK) == -1) {
	FILE *file = fopen("OpenMP.csv", "a");
	fprintf(file, "Matrix,NonZeros, ,MaxErrorCSR,MaxErrorELL, ,serialCSR,serialEll");
	for (int i = 0; i < MAX_THREADS; i++)
	{
		fprintf(file, ",CSR(%d),ELL(%d)", i + 1, i + 1);
	}
	fprintf(file, "\n");
	fclose(file);
	}
}


void computeOpenMPCalculations(char const *matrix) {
	
	Matrix m(matrix);
	if (m.failed)
		return;

	double *x = (double *)malloc(m.N * sizeof(double));
	srand(time(0));
	for (int i = 0; i < m.N; i++)
	{
		x[i] = 100.0f * static_cast<float>(rand()) / RAND_MAX;
	}
	double *csrResultParallelOMP = (double *)malloc(m.M * sizeof(double));
	double *csrResultSerial = (double *)malloc(m.M * sizeof(double));
	double *ellResultParallelOMP = (double *)malloc(m.M * sizeof(double));
	double *ellResultSerial = (double *)malloc(m.M * sizeof(double));
	double ompCSRFlops[MAX_THREADS];
	m.convertMatrixToCSR();

	struct timeval t1, t2;
	//Open mp
	for (int threads = 1; threads <= MAX_THREADS; threads++)
	{
		double parallelCSRaverageOMP = 0.0;
		for (int i = 0; i < 20; i++)
		{
				gettimeofday(&t1, NULL);
			matrixVectorCSROpenMP(m.IRP, m.JA, m.AS, x, m.M, threads, csrResultParallelOMP);
			gettimeofday(&t2, NULL);
			parallelCSRaverageOMP += (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_usec - t1.tv_usec) * 1000.0;
		}
		parallelCSRaverageOMP /= 20.0;
		ompCSRFlops[threads - 1] = (2.0 * m.nz) / (parallelCSRaverageOMP);
	}

	//Serial
	double serialCSRaverage = 0.0;
	for (int i = 0; i < 20; i++)
	{
		gettimeofday(&t1, NULL);
		serialMMCSR(m.IRP, m.JA, m.AS, x, m.M, csrResultSerial);
		gettimeofday(&t2, NULL);
		serialCSRaverage += (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_usec - t1.tv_usec) * 1000.0;
	}
	serialCSRaverage /= 20.0;

	//ELLPACK
	try
	{
		m.convertMatrixToELLPACK();

		//Serial
		double serialELLaverage = 0.0;
		for (int i = 0; i < 20; i++)
		{
			gettimeofday(&t1, NULL);
			serialMMELLPACK(m.maxNZ, m.JA, m.AS, x, m.M, ellResultSerial);
			gettimeofday(&t2, NULL);
			serialELLaverage += (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_usec - t1.tv_usec) * 1000.0;
		}
		serialELLaverage /= 20.0;

		//OpenMP

		double ompELLFlops[MAX_THREADS];
		for (int threads = 1; threads <= MAX_THREADS; threads++)
		{
			double parallelELLaverageOMP = 0.0;
			for (int i = 0; i < 20; i++)
			{
				gettimeofday(&t1, NULL);
				matrixVectorELLOpenMP(m.maxNZ, m.JA, m.AS, x, m.M, threads, ellResultParallelOMP);
				gettimeofday(&t2, NULL);
				parallelELLaverageOMP += (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_usec - t1.tv_usec) * 1000.0;
			}
			parallelELLaverageOMP /= 20.0;
			ompELLFlops[threads - 1] = (2.0 * m.nz) / (parallelELLaverageOMP);
		}

		double serialCsrFlops, serialEllFlops;
		serialCsrFlops = (2.0 * m.nz) / (serialCSRaverage);
		serialEllFlops = (2.0 * m.nz) / (serialELLaverage);

		double maxDiffCSROmp = 0.0, maxDiffELLOmp = 0.0;
		double diffCSROmp, diffELLOmp;
		int count = 0;
		for (int i = 0; i < m.M; i++)
		{
			diffCSROmp = abs(csrResultParallelOMP[i] - csrResultSerial[i]) / csrResultSerial[i];
			diffELLOmp = abs(ellResultParallelOMP[i] - ellResultSerial[i]) / ellResultSerial[i];

			maxDiffCSROmp = diffCSROmp > maxDiffCSROmp ? diffCSROmp : maxDiffCSROmp;
			maxDiffELLOmp = diffELLOmp > maxDiffELLOmp ? diffELLOmp : maxDiffELLOmp;
		}
		writeToFileOpenMP(serialCsrFlops, ompCSRFlops, serialEllFlops, ompELLFlops, matrix, MAX_THREADS, m.nz, maxDiffCSROmp, maxDiffELLOmp);
		free(ellResultParallelOMP);
		free(ellResultSerial);
	}
	catch (const std::exception&)
	{
		double serialCsrFlops;
		serialCsrFlops = (2.0 * m.nz) / (serialCSRaverage);

		double maxDiffCSROmp = 0.0;
		double diffCSROmp;
		int count = 0;
		for (int i = 0; i < m.M; i++)
		{
			diffCSROmp = abs(csrResultParallelOMP[i] - csrResultSerial[i]) / csrResultSerial[i];

			maxDiffCSROmp = diffCSROmp > maxDiffCSROmp ? diffCSROmp : maxDiffCSROmp;
		}

		cout << "Matrix " << matrix << " too large for ellpack" << endl;
		writeToFileOpenMP(serialCsrFlops, ompCSRFlops, -1, NULL, matrix, MAX_THREADS, m.nz, maxDiffCSROmp, -1);
	}


	free(csrResultParallelOMP);
	free(csrResultSerial);	
}

void computeCUDACalculations(char const *matrix) {
	Matrix m(matrix);
	if (m.failed)
		return;

	double *x = (double *)malloc(m.N * sizeof(double));
	srand(time(0));
	for (int i = 0; i < m.N; i++)
	{
		x[i] = 100.0f * static_cast<float>(rand()) / RAND_MAX;
	}

	double *csrResultParallelCuda;
	double *csrResultSerial = (double *)malloc(m.M * sizeof(double));
	double *ellResultParallelCuda;
	double *ellResultSerial = (double *)malloc(m.M * sizeof(double));

	m.convertMatrixToCSR();

	struct timeval t1, t2;
	//Cuda
	double parallelCSRaverageCuda;
	csrResultParallelCuda = matrixVectorCSRCUDA(m.IRP, m.JA, m.AS, x, m.M, m.N, m.nz, 20, &parallelCSRaverageCuda);

	//Serial
	double serialCSRaverage = 0.0;
	for (int i = 0; i < 20; i++)
	{
			gettimeofday(&t1, NULL);
		serialMMCSR(m.IRP, m.JA, m.AS, x, m.M, csrResultSerial);
		gettimeofday(&t2, NULL);
		serialCSRaverage += (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_usec - t1.tv_usec) * 1000.0;

	}
	serialCSRaverage /= 20.0;

	try
	{
		//ELLPACK
		m.convertMatrixToELLPACK();

		//Serial
		double serialELLaverage = 0.0;
		for (int i = 0; i < 20; i++)
		{
			gettimeofday(&t1, NULL);
			serialMMELLPACK(m.maxNZ, m.JA, m.AS, x, m.M, ellResultSerial);
				gettimeofday(&t2, NULL);
			serialELLaverage += (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_usec - t1.tv_usec) * 1000.0;
		}
		serialELLaverage /= 20.0;

		//CUDA
		double parallelEllaverageCuda;
		ellResultParallelCuda = matrixVectorELLCUDA(m.maxNZ, m.JA, m.AS, x, m.M, m.N, m.nz, 20, &parallelEllaverageCuda);
		double serialCsrFlops, serialEllFlops, parallelCsrFlops, parallelEllFlops;

		serialCsrFlops = (2.0 * m.nz) / (serialCSRaverage);
		serialEllFlops = (2.0 * m.nz) / (serialELLaverage);
		parallelCsrFlops = (2.0 * m.nz) / (parallelCSRaverageCuda * 1000000.0);
		parallelEllFlops = (2.0 * m.nz) / (parallelEllaverageCuda * 1000000.0);

		double maxDiffCSRCuda = 0.0, maxDiffELLCuda = 0.0;
		double diffCSRCuda, diffELLCuda;
		int count = 0;
		for (int i = 0; i < m.M; i++)
		{
			diffCSRCuda = abs(csrResultParallelCuda[i] - csrResultSerial[i]);
			diffELLCuda = abs(ellResultParallelCuda[i] - ellResultSerial[i]);

			maxDiffCSRCuda = diffCSRCuda > maxDiffCSRCuda ? diffCSRCuda : maxDiffCSRCuda;
			maxDiffELLCuda = diffELLCuda > maxDiffELLCuda ? diffELLCuda : maxDiffELLCuda;
		}

		writeToFileCUDA(serialCsrFlops, parallelCsrFlops, serialEllFlops, parallelEllFlops, matrix, m.nz, maxDiffCSRCuda, maxDiffELLCuda);
		free(ellResultSerial);
	}
	catch (const std::exception&)
	{
		cout << "Matrix " << matrix << " too large for ellpack" << endl;

		double serialCsrFlops, parallelCsrFlops;
		serialCsrFlops = (2.0 * m.nz) / (serialCSRaverage);
		parallelCsrFlops = (2.0 * m.nz) / (parallelCSRaverageCuda * 1000000.0);

		double maxDiffCSRCuda = 0.0;
		double diffCSRCuda;
		int count = 0;
		for (int i = 0; i < m.M; i++)
		{
			diffCSRCuda = abs(csrResultParallelCuda[i] - csrResultSerial[i]);

			maxDiffCSRCuda = diffCSRCuda > maxDiffCSRCuda ? diffCSRCuda : maxDiffCSRCuda;
		}

		writeToFileCUDA(serialCsrFlops, parallelCsrFlops, -1, -1, matrix, m.nz, maxDiffCSRCuda, -1);
	}
	
	free(csrResultSerial);	
}

int main(int argc, char *argv[])
{
	bool singleMatrixText = false;
	//Create output files
	if (argc < 1) {
		printf("Select running mode: CUDA|OpenMP\n");
		exit(1);
	}
	bool cuda = false;
	string mode(argv[1]);
	if (mode == "CUDA") {
		createFileCuda();
		cuda = true;
	}
	else
		createFileOpenMP();

	//Open files in directory
	if (singleMatrixText)
		if (cuda)
			computeCUDACalculations("dc1");
		else
			computeOpenMPCalculations("dc1");
	else {
		DIR *dir;
		struct dirent *ent;
		if ((dir = opendir("Matrices")) != NULL) {
			/* print all the files and directories within directory */
			int i = 0;
			while ((ent = readdir(dir)) != NULL) {
				if (strcmp(ent->d_name, "ZIPs") != 0 && strcmp(ent->d_name, ".") != 0
					&& strcmp(ent->d_name, "..") != 0) {
					if (cuda) {
						computeCUDACalculations(ent->d_name);
					}
					else
						computeOpenMPCalculations(ent->d_name);
				}
			}
			closedir(dir);
		}
		else {
			/* could not open directory */
			perror("");
			return EXIT_FAILURE;
		}
	}

	return 0;
}
