#include "Matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <string>



Matrix::Matrix(char const *matrix) {

	DEATH_MATRIX1 = "webbase-1M";
	DEATH_MATRIX2 = "dc1";
	this->matrix = matrix;
	char fileName[50];
	sprintf(fileName, "Matrices/%s/%s.mtx", matrix, matrix);

	if ((f = fopen(fileName, "r")) == NULL)
		exit(1);

	if (mm_read_banner(f, &matcode) != 0)
	{
		printf("Could not process Matrix Market banner.\n");
		exit(1);
	}

	if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
		mm_is_sparse(matcode))
	{
		printf("Sorry, this application does not support ");
		printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
		exit(1);
	}
	failed = false;
	if (mm_is_array(matcode)) {
		failed = true;
		return;
	}

	symetric = false;
	pattern = false;
	if ((mm_is_symmetric(matcode) || mm_is_skew(matcode)) && mm_is_sparse(matcode)) {
		symetric = true;
	}

	if ((mm_is_pattern(matcode)))
		pattern = true;
	/* find out size of sparse matrix .... */

	if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0)
		exit(1);


	/* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
	/*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
	/*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
	int i, row, col, nonDiagonalElements = 0;
	double val;
	for (i = 0; i < nz; i++)
	{
		if (!pattern)
		{
			fscanf(f, "%d %d %lg\n", &row, &col, &val);
		}
		else
		{
			fscanf(f, "%d %d\n", &row, &col);
			val = 1.0;
		}
		row--;  /* adjust from 1-based to 0-based */
		col--;
		rows.push_back(row);
		columns.push_back(col);
		vals.push_back(val);
		if (symetric) {
			if (row != col) {
				rows.push_back(col);
				columns.push_back(row);
				vals.push_back(val);
				nonDiagonalElements++;
			}
		}
	}
	nz += nonDiagonalElements;

	if (f != stdin) fclose(f);

	/************************/
	/* now write out matrix */
	/************************/

	mm_write_banner(stdout, matcode);

	mm_write_mtx_crd_size(stdout, M, N, (nz - nonDiagonalElements));


	/************************/
	/* Prepocess the matrix */
	/************************/

	// First we need to sort the matrix by rows. Merge sort for now.
	sortByRows();
}

Matrix::~Matrix()
{
	if (format == CSR)
		free(IRP);
	else {
		free(JA);
		free(AS);
	}
	rows.clear();
	columns.clear();
	vals.clear();
}

void Matrix::swap(int i1, int i2)
{
	int temp;

	temp = rows[i1];
	rows[i1] = rows[i2];
	rows[i2] = temp;

	temp = columns[i1];
	columns[i1] = columns[i2];
	columns[i2] = temp;

	double tempD;
	tempD = vals[i1];
	vals[i1] = vals[i2];
	vals[i2] = tempD;
}

void Matrix::sortByRows()
{
	int *B = (int *)malloc(nz * sizeof(int));
	int *C = (int *)malloc(nz * sizeof(int));
	double *D = (double *)malloc(nz * sizeof(double));
	BottomUpMergeSort(B, nz, C, D);
	free(B);
	free(C);
	free(D);
}

//  Left run is A[iLeft :iRight-1].
// Right run is A[iRight:iEnd-1  ].
void Matrix::BottomUpMerge(int iLeft, int iRight, int iEnd, int B[], int C[], double D[])
{
	int i = iLeft, j = iRight;
	// While there are elements in the left or right runs...
	for (int k = iLeft; k < iEnd; k++) {
		// If left run head exists and is <= existing right run head.
		if (i < iRight && (j >= iEnd || rows[i] < rows[j])) {
			B[k] = rows[i];
			C[k] = columns[i];
			D[k] = vals[i];
			i = i + 1;
		}
		else if (i < iRight && (j >= iEnd || rows[i] == rows[j])) {
			if (columns[i] <= columns[j]) {
				B[k] = rows[i];
				C[k] = columns[i];
				D[k] = vals[i];
				i = i + 1;
			}
			else {
				B[k] = rows[j];
				C[k] = columns[j];
				D[k] = vals[j];
				j = j + 1;
			}
		}
		else {
			B[k] = rows[j];
			C[k] = columns[j];
			D[k] = vals[j];
			j = j + 1;
		}
	}
}

// array A[] has the items to sort; array B[] is a work array
void Matrix::BottomUpMergeSort(int B[], int n, int C[], double D[])
{
	// Each 1-element run in A is already "sorted".
	// Make successively longer sorted runs of length 2, 4, 8, 16... until whole array is sorted.
	for (int width = 1; width < n; width = 2 * width)
	{
		// Array A is full of runs of length width.
		for (int i = 0; i < n; i = i + 2 * width)
		{
			// Merge two runs: A[i:i+width-1] and A[i+width:i+2*width-1] to B[]
			// or copy A[i:n-1] to B[] ( if(i+width >= n) )
			BottomUpMerge(i, std::min(i + width, n), std::min(i + 2 * width, n), B, C, D);
		}
		// Now work array B is full of runs of length 2*width.
		// Copy array B to array A for next iteration.
		// A more efficient implementation would swap the roles of A and B.
		CopyArray(B, n, C, D);
		// Now array A is full of runs of length 2*width.
	}
}

void Matrix::CopyArray(int B[], int n, int C[], double D[])
{
	for (int i = 0; i < n; i++) {
		rows[i] = B[i];
		columns[i] = C[i];
		vals[i] = D[i];
	}
}

void Matrix::convertMatrixToCSR()
{
	format = CSR;
	// At the end of the sort, cols should be equal to JA and vals to AS
	JA = columns.data();
	AS = vals.data();


	// We now need to calculate IRP. To do this we simply need to count the number of elements in each row and add that value to the previous one.
	// Remember that the rows array is already sorted
	int i;
	IRP = (int *)malloc((M + 1) * sizeof(int));
	IRP[0] = 0;
	IRP[M] = nz;
	int j;
	for (i = 1; i < M; i++)
	{
		j = IRP[i - 1];
		IRP[i] = IRP[i - 1];
		while (rows[j] == i - 1)
		{
			IRP[i]++;
			j++;
		}
	}
}

void Matrix::convertMatrixToELLPACK()
{
	std::string death(matrix);
	if (death == DEATH_MATRIX1 || death == DEATH_MATRIX2) {
		throw std::bad_alloc();
	}
	format = ELLPACK;
	// To simply the calculations, the matrix was sorted by rows
	// We now need to find the maximum number of nonzeros per row
	int max = 0, i, count = 0, row = 0;
	for (i = 0; i < nz; i++)
	{
		if (rows[i] == row)
			count++;
		else
		{
			max = count > max ? count : max;
			count = 1;
			row++;
		}
	}
	max = count > max ? count : max;
	
	//JA and AS were previously allocated in the CSR function, reallocation is needed

	JA = (int *)malloc(max * M * sizeof(int));
	AS = (double *)malloc(max * M * sizeof(double));
	if (!JA || !AS) {
		throw std::bad_alloc();
	}

	int j = 0, temp = 0;
	int index;
	for (i = 0; i < M; i++)
	{
		index = i * max;
		for (j = 0; j < max; j++)
		{
			if (temp < nz)
			{
				if (rows[temp] == i)
				{
					JA[index] = columns[temp];
					AS[index] = vals[temp];
				}
				else
				{
					JA[index] = JA[index - 1];
					AS[index] = 0;
					temp--;
				}

			}
			else
			{
				JA[index] = JA[index - 1];
				AS[index] = 0;
				
			}
			temp++;
			index++;
		}
	}

	maxNZ = max;
}

