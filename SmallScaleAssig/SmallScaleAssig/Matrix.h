#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>

extern "C" {
#include "mmio.h"
}

enum MATRIX_FORMAT {
	CSR,
	ELLPACK
};

class Matrix {

public:

	Matrix(char const *matrix);
	~Matrix();
	
	void convertMatrixToCSR();
	void convertMatrixToELLPACK();

	int *JA;
	double *AS;
	int maxNZ;
	int *IRP;
	int M, N, nz;

	bool failed;
	std::string matrix;
private:
	std::vector<int> rows;
	std::vector<int> columns;
	std::vector<double> vals;

	void BottomUpMergeSort(int B[], int n, int C[], double D[]);
	void swap(int i1, int i2);
	void sortByRows();

	MM_typecode matcode;
	int ret_code;
	FILE *f;
	MATRIX_FORMAT format;
	bool symetric, pattern;

	void BottomUpMerge(int iLeft, int iRight, int iEnd, int B[], int C[], double D[]);
	void CopyArray(int B[],int n, int C[], double D[]);	

	std::string DEATH_MATRIX1;
	std::string DEATH_MATRIX2;
};
