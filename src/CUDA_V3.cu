#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <chrono>

// constant definition
#define MOMENTS_PER__BLOCK (107)
#define MOMENTS_PER_THREAD 5
#define ZERO_THRESHOLD 1e-6
#define MASK_SIZE 5

// function declaration 
__device__ int updateSpin(int* G, float* w, int x, int y, int n);
__global__ void isingOnce(int* G, int* R, float* w, int n);
__device__ int posRoll(int i, int j, int n);
__device__ void readElements(int *g_mem, int *s_mem, int g_index, int s_index, int elements);
void readFile(const char *file_name, int *Table, int n);
template <class T> void swap(T **x, T **y);
bool validation(int *Result, int *Solution, int n);


static const char *RESULT_STATUS[] = { "WRONG", "CORRECT" };


// Declaration of the weights consisting the window mask
const float M[(MASK_SIZE * MASK_SIZE)] = {
	0.004,  0.016,  0.026,  0.016,   0.004,
	0.016,  0.071,  0.117,  0.071,   0.016,
	0.026,  0.117,  0.0,    0.117,   0.026,
	0.016,  0.071,  0.117,  0.071,   0.016,
	0.004,  0.016,  0.026,  0.016,   0.004
};


int main(int argc, char *argv[]) {
	int *A, *B, *d_A, *d_R;
	int n = 517;
	int k = 11;
	float* W;

	size_t bytes_of_model = n * n * sizeof(int);
	size_t bytes_of_weight = (MASK_SIZE * MASK_SIZE) * sizeof(float);


	// allocates memory for the model's initial state and also the final solution after k updates
	// then we import the data from the .bin files
	A = (int*)malloc(bytes_of_model);
	B = (int*)malloc(bytes_of_model);
	readFile("conf-init.bin", A, 517);
	readFile("conf-11.bin", B, 517);



	// Alocates memory in the device for the model(d_A) the result after
	// the k-updates updated_model(d_R) and the window mask matrix(W)
	cudaMalloc(&d_A, bytes_of_model);
	cudaMalloc(&d_R, bytes_of_model);
	cudaMalloc(&W, bytes_of_weight);

	// Copy the model and mask matrices from host to device while measuring the time
	auto start_comm = std::chrono::high_resolution_clock::now();
	cudaMemcpy(d_A, A, bytes_of_model, cudaMemcpyHostToDevice);
	cudaMemcpy(W, M, bytes_of_weight, cudaMemcpyHostToDevice);
	auto finish_comm = std::chrono::high_resolution_clock::now();

	// set the dimensions of the thread per block and blocks in the grid
	dim3 dimBlock(1, 128);
	dim3 dimGrid(n, 1);
	int shared_mem = (517 + 4) * 5 * sizeof(int);


	// Performs k-updates while measuring the time of execution and for any errors that occured
	auto start = std::chrono::high_resolution_clock::now();
	for (int t = 0;t < k;t++) {
		isingOnce << <dimGrid, dimBlock, shared_mem >> > (d_A, d_R, W, n);
		swap(&d_A, &d_R);
	}
	cudaError_t errSync = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess)
		printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess)
		printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
	auto finish = std::chrono::high_resolution_clock::now();



	// Copy the result of the calculation from the device to the host
	auto comm_time = finish_comm - start_comm;
	start_comm = std::chrono::high_resolution_clock::now();
	cudaMemcpy(A, d_A, bytes_of_model, cudaMemcpyDeviceToHost);
	finish_comm = std::chrono::high_resolution_clock::now();
	comm_time = comm_time + finish_comm - start_comm;


	// Print results & check validation
	std::cout << "Execution time is: " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() << " ns\n";
	std::cout << "Communication time is: " << std::chrono::duration_cast<std::chrono::nanoseconds>(comm_time).count() << " ns\n";
	std::cout << "Result is: " << RESULT_STATUS[validation(A, B, n)] << std::endl;


	// Free dynamic memory stored on both the host and the device
	cudaFree(d_A);
	cudaFree(d_R);
	cudaFree(W);
	free(A);
	free(B);

	// Waits so the console does not exit and the results are visible
	getchar();

	return 0;

}


/*
	This function calculates as single evolution of the ising model
	It allocates the necessary shared memory for the operations.
	Then reads the elements of a row of the W matrix which is assigned to this block
	alongside with the 2 adjacent row (upper & lower).

	Waits for the operation to end - (local synchronazation in a block has a small cost)

	because in order to fill the boundary conditions described in the pdf we have to
	perform a constant amount of copy operations in shared memory we manually assign them
	to threads in order to avoid bank conflicts.

	Also transfers the w mask matrix from global to shared

	Once a again we wait for the operation to finish

	and we proceed in calculating the result for each element in our row and store it
	in the correct position in the R (result matrix) again with coalescing memory access
	all the calculating operations in this point are done in shared memory
*/
__global__ void isingOnce(int* G, int* R, float* w, int n) {
	extern __shared__ int S[];
	__shared__ float s_w[25];

	unsigned int begin_row = blockIdx.x * n / gridDim.x;
	unsigned int end_row = (blockIdx.x + 1) * n / (gridDim.x);
	unsigned int elements = (end_row - begin_row) * n;


	for (int row = -2; row <= 2; row++) {
		unsigned int s_row = 2 + (row + 2)*(n + 4);
		unsigned int g_row = ((blockIdx.x + gridDim.x - row) % gridDim.x) * n;
		readElements(G, S, g_row, s_row, n);
	}
	//skata apo do kai kato

	__syncthreads();

	int me = threadIdx.x * blockDim.y + threadIdx.y;
	int new_n = n + 4;
	int offset = 2;

	if (me < 25 && me >= 0) {
		s_w[me] = w[me];
	}
	else {
		switch (me) {
		case 25:
			S[2 * new_n] = S[2 * new_n + n];
			break;
		case 26:
			S[2 * new_n + 1] = S[2 * new_n + n + 1];
			break;
		case 27:
			S[2 * new_n + n + 2] = S[2 * new_n + 2];
			break;
		case 28:
			S[2 * new_n + n + 3] = S[2 * new_n + 3];
			break;
		case 29:
			S[0] = S[n];
			break;
		case 30:
			S[1] = S[n + 1];
			break;
		case 31:
			S[new_n] = S[new_n + n];
			break;
		case 32:
			S[new_n + 1] = S[new_n + n + 1];
			break;
		case 33:
			S[n + 2] = S[2];
			break;
		case 34:
			S[n + 3] = S[3];
			break;
		case 35:
			S[new_n + n + 2] = S[new_n + 2];
			break;
		case 36:
			S[new_n + n + 3] = S[new_n + 3];
			break;
		case 37:
			S[(4) * new_n] = S[(4) * new_n + n];
			break;
		case 38:
			S[(4) * new_n + 1] = S[(4) * new_n + n + 1];
			break;
		case 39:
			S[(3) * new_n] = S[(3) * new_n + n];
			break;
		case 40:
			S[(3) * new_n + 1] = S[(3) * new_n + n + 1];
			break;
		case 41:
			S[(4) * new_n + n + 2] = S[(4) * new_n + 2];
			break;
		case 42:
			S[(4) * new_n + n + 3] = S[(4) * new_n + 3];
			break;
		case 43:
			S[(3) * new_n + n + 2] = S[(3) * new_n + 2];
			break;
		case 44:
			S[(3) * new_n + n + 3] = S[(3) * new_n + 3];
			break;
		default:
			break;
		}

	}

	__syncthreads();
	int thread_offset = blockDim.x * blockDim.y;
	//komple
	for (int i = 0; i <= elements / thread_offset; i++) {
		int pos = me + i * thread_offset;
		if (pos < elements) {
			R[blockIdx.x * n + pos] = updateSpin(S, s_w, 2, 2 + pos, n + 4);
		}
	}
}


/*
	Function that runs on the device
	It reads from global memory "elements" element starting from "g_index"
	and	stores them in shared memory in the index position "s_index"

	To perform the reading operation from the slow global memory as efficiently
	as possible it calculates the number of total threads in this block and performs
	as coalescing memory access meaning threads are reading continues memory addresses

	if the division between the number of element to read and the threads of the block isnt
	perfect some threads will wait but still it is more efficient than splitting the elements
	to read in blocks and then assign each block a thread

	more on this:
	https://devblogs.nvidia.com/how-access-global-memory-efficiently-cuda-c-kernels/
*/
__device__ void readElements(int *g_mem, int *s_mem, int g_index, int s_index, int elements) {
	unsigned int me = threadIdx.x * blockDim.y + threadIdx.y;
	unsigned int offset = blockDim.x * blockDim.y;
	unsigned int iterations = elements / offset;

	for (int i = 0;i <= iterations; i++) {
		unsigned int pos = me + i * offset;
		if (pos < elements) {
			s_mem[s_index + pos] = g_mem[g_index + pos];
		}
	}
}


/*
	Function that runs on the device
	Updates the value of  the elements at position (x*n + y) of the "G" matrix
	by calculating the weighted average of its neighbors according to the weights
	given by the "w" matrix

	Then depending on the result it classifies it as (-1) if is enough negative, (+1) if is enough positive
	and as unchanged if is zero depending of the "ZERO_THRESHOLD" constant.

	"posRoll" is called in order to assure thay we access the right index in the "G" matrix as we iterate
	in the element's neighborhood.
*/
__device__ int updateSpin(int* G, float* w, int x, int y, int n) {
	float sum = 0.0;

	for (int i = 0; i < MASK_SIZE; i++) {
		for (int j = 0; j < MASK_SIZE; j++) {
			sum = sum + w[(i * 5 + j)] * G[posRoll((x + i - 2), (y - j + 2), n)];
		}
	}

	if (fabs(sum) < ZERO_THRESHOLD) {
		return G[x * n + y];
	}
	else if (sum > 0) {
		return 1;
	}
	else {
		return -1;
	}
}


/*
	Given the two index positions of an element of a n-by-n matrix first performs a circular shift of the element's coordinates
	as to satisfy the toroidal boundary conditions and then returns the index of the row major equivilant of this mattrix

	for instance suppose we have and 3-by-3 matrix and want to access the (-1,-1) -> (2 ,2) -> (8)
*/
__device__ int posRoll(int i, int j, int n) {
	int l = (i + n) % n;
	int k = (j + n) % n;

	return (l * n + k);
}


// Reads the information from a bin folder and stores it in "Table" of size n-by-n
void readFile(const char *file_name, int *Table, int n)
{
	FILE* fptr = fopen(file_name, "r");
	if (fptr == NULL) {
		printf("Could not opent he file. Exiting...\n");
		exit(-1);
	}
	else {
		for (int i = 0;i < n; i++)
			for (int j = 0;j < n; j++)
				fread(&Table[i * n + j], sizeof(int), 1, fptr);
	}
}


// Compares two Matrices of size n-by-n element by element and returns true is case 
// of perfect match or false in case of one or more mismatches between their elements
bool validation(int * Result, int * Solution, int n) {
	int result = true;
	for (int i = 0;i < n;i++) {
		for (int j = 0;j < n;j++) {
			if (fabs(Result[i * n + j] - Solution[i * n + j]) > ZERO_THRESHOLD) {
				std::cout << i << ", " << j << " " << Result[i * n + j] << " " << Solution[i * n + j] << std::endl;
				result = false;
				return result;
			}
		}
	}
	return result;
}


// Switch to abstract pointers
template <class T>
void swap(T **x, T **y) {
	T *temp = *x;
	*x = *y;
	*y = temp;
}


