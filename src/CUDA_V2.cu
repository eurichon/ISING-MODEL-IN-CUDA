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
#include <atomic>
#define WEIGHT_SIZE 5
#define MOMENTS_PER_THREAD 5
#define MOMENTS_PER__BLOCK 107

//void ising(int* G, int* R, float* w, int k, int n);

__global__ void isingOnce(int* G, int* R, float* w, int n);
__device__ int updateSpin(int* G, float* w, int x, int y, int n);

__device__ int posRoll(int i, int j, int n);

int main() {





	int n = 517;
	size_t bytes = n * n * sizeof(int);
	float  M[(WEIGHT_SIZE * WEIGHT_SIZE)] = {
	0.004,  0.016,  0.026,  0.016,   0.004,
	0.016,  0.071,  0.117,  0.071,   0.016,
	0.026,  0.117,  0.0,    0.117,   0.026,
	0.016,  0.071,  0.117,  0.071,   0.016,
	0.004,  0.016,  0.026,  0.016,   0.004 };


	int* A = (int*)malloc(n * n * sizeof(int));
	int* B = (int*)malloc(n * n * sizeof(int));





	FILE* fptr;


	fptr = fopen("conf-init.bin", "r");
	if (fptr == NULL) {
		printf("Could not opent he file. Exiting...\n");
		exit(-1);
	}
	else {
		for (int i = 0;i < n; i++)
			for (int j = 0;j < n; j++)
				fread(&A[i * n + j], sizeof(int), 1, fptr);
	}

	fptr = fopen("conf-4.bin", "r");
	if (fptr == NULL) {
		printf("Could not opent he file. Exiting...\n");
		exit(-1);
	}
	else {
		for (int i = 0;i < n; i++)
			for (int j = 0;j < n; j++)
				fread(&B[i * n + j], sizeof(int), 1, fptr);
	}


	int* d_A;
	int* d_R;
	float* W;
	// Allocation memory for these pointers
	cudaMalloc(&d_A, bytes);
	cudaMalloc(&W, (WEIGHT_SIZE * WEIGHT_SIZE) * sizeof(float));
	cudaMalloc(&d_R, bytes);

	//Copy Data From CPU to GPU memory
	cudaMemcpy(d_A, A, bytes,
		cudaMemcpyHostToDevice);
	cudaMemcpy(W, M, (WEIGHT_SIZE * WEIGHT_SIZE) * sizeof(float),
		cudaMemcpyHostToDevice);

	int k = 4;




	//ising(d_A, d_R, W, k, n);

	int BLOCK_SIZE = 1 << 5;

	// Blocks per Grid
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);


	// Appropirate Grid dimmensions for any n .
	// There are cases where n mod BLOCK_Size != 0
	int bx = (n / MOMENTS_PER__BLOCK + dimBlock.x - 1) / dimBlock.x;
	int by = (n / MOMENTS_PER__BLOCK + dimBlock.y - 1) / dimBlock.y;

	dim3 dimGrid(bx, by);



	// call kernel k times
	for (int t = 0;t < k;t++) {
		isingOnce << <dimGrid, dimBlock >> > (d_A, d_R, W, n);

		// Wait for GPU Threads to finish
		//cudaDeviceSynchronize();
		// Swap Matrixes and Continue looping
		int* temp = d_A;
		d_A = d_R;
		d_R = temp;
	}
	cudaDeviceSynchronize();
	cudaMemcpy(A, d_A, bytes,
		cudaMemcpyDeviceToHost);


 // Check reults
	int result = 1;
	for (int i = 0;i < n;i++) {
		for (int j = 0;j < n;j++) {
			if (B[i * n + j] != A[i * n + j]) {
				result = 0;
				//printf("A  = %d B = %d \t i = %d , j = %d\n", A[i * n + j], B[i * n + j], i, j);
				break;
			}
		}
	}

	printf("\n");

	(result) ? (printf("Success\n")) : printf("Fail\n");




	free(A);
	free(B);

	cudaFree(d_A);
	cudaFree(d_R);




	return 0;
}



// This kernel executes only one iteration of the ising model
// It loops through all the points and writes the updated spins in an intermidiate matrix.
__global__ void isingOnce(int* G, int* R, float* w, int n) {
// Index of i and j for 2D Grid and 2D blocks
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;


	// Set Begining and ending points for the block of moments that the thread is
	//responsible for.
	int begin_x = n / blockDim.x * x;
	int begin_y = n / blockDim.y * y;
	int end_x = (x == (blockDim.x - 1)) ? (n / blockDim.x * (x + 1) + n % blockDim.x) : (n / blockDim.x * (x + 1));
	int end_y = (y == (blockDim.y - 1)) ? (n / blockDim.y * (y + 1) + n % blockDim.y) : (n / blockDim.y * (y + 1));

	for (int i = begin_x; i < end_x;i++) {
		for (int j = begin_y; j < end_y;j++) {
			//Make sure to be in bounds
			if (i < n && j < n) {
				int pos = i * n + j;
				R[pos] = updateSpin(G, w, i, j, n);
			}
		}
	}


}


//This function updates the spin of a single moment.
//The Grid Matrix G is indexed correctly with the use of posRoll.
//The functions returns the updated spin value.
//This function is called and executed from the device.
__device__ int updateSpin(int* G, float* w, int x, int y, int n) {
	float sum = 0.0;
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			sum = sum + w[(i * 5 + j)] * G[posRoll((x + i - 2), (y - j + 2), n)];
		}
	}

	if (sum < 0.0000001 && sum > -0.0000001) {
		return G[x * n + y];
	}
	else if (sum > 0) {
		return 1;
	}
	else {
		return -1;
	}

}

//This function rolls 2D coordinates around the matrix
//and returns the corresponding 1D index.
//This function is called and executed from the device
__device__ int posRoll(int i, int j, int n) {
	//Roll coordinates
	int l = (i + n) % n;
	int k = (j + n) % n;
	
	return (l * n + k);
}
