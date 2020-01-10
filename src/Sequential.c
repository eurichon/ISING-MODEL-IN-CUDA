#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#define SPIN_SIZE 517
#define WEIGHT_SIZE 5
#define SIGN(X) ((X > 0)?(1):(-1))

void ising(int *G,double *w, int k, int n);
void isingOnce(int *G, int *R, double *w, int n);
int updateSpin(int *G, double *w, int x, int y, int n);
int posRoll(int i, int j, int n);
void printArray(int *arr,int i,int j,int n);

int main(int argc, char *argv[]){
    FILE *fptr;
	// int A[(SPIN_SIZE * SPIN_SIZE)],
	// 	B[(SPIN_SIZE * SPIN_SIZE)],
	// 	R[(SPIN_SIZE * SPIN_SIZE)];

  int *A = (int*)malloc(SPIN_SIZE*SPIN_SIZE*sizeof(int));
  int *B = (int*)malloc(SPIN_SIZE*SPIN_SIZE*sizeof(int));

	//Define W
	double M[(WEIGHT_SIZE * WEIGHT_SIZE)] = {
		0.004,  0.016,  0.026,  0.016,   0.004,
		0.016,  0.071,  0.117,  0.071,   0.016,
		0.026,  0.117,  0.0,    0.117,   0.026,
		0.016,  0.071,  0.117,  0.071,   0.016,
		0.004,  0.016,  0.026,  0.016,   0.004};

	int k = 1;

	printf("Here ig got\n");
	double sum = 0.0;
	for(int i = 0;i < 25;i++){
		sum = sum + M[i];
	}
	printf("sum is: %lf \n",sum);
	// Read the files
    fptr = fopen("conf-init.bin","r");
	if(fptr == NULL){
		printf("Could not opent he file. Exiting...\n");
		exit(-1);
	}else{
		for (int i =0;i < SPIN_SIZE; i++)
			for(int j =0;j < SPIN_SIZE; j++)
				fread(&A[i * SPIN_SIZE + j], sizeof(int), 1, fptr);
	}

	fptr = fopen("conf-1.bin","r");
	if(fptr == NULL){
		printf("Could not opent he file. Exiting...\n");
		exit(-1);
	}else{
		for (int i =0;i < SPIN_SIZE; i++)
			for(int j =0;j < SPIN_SIZE; j++)
				fread(&B[i * SPIN_SIZE + j], sizeof(int), 1, fptr);
	}


	//Execute Ising Model
	int n = SPIN_SIZE;
	ising(A,  M, k, n);



	// Check if the results match up with the given file.
	int result = 1;
	for(int i =0;i< n;i++){
		for(int j = 0;j < n;j++){
			if(B[i*n+j] != A[i*n+j]){
				result = 0;
			}
		}
	}

	printf("\n");

	(result)?(printf("Success\n")):printf("Fail\n");


	printf("Press any key to ESC!\n");
	getchar();
	return 0;
}

void printArray(int *arr,int x,int y,int n){
	for(int i =0;i<x;i++){
		printf("\n");
			for(int j=0;j<y;j++)
				printf("%d, ",arr[i * n + j]);
	}
	printf("\n");
}

//This function executes the ising model algorithm k times.
//After each iteration the result is writen from the R matrix to the G.
void ising(int *G, double *w, int k, int n){

  int *R = (int*)malloc(n*n*sizeof(int));
  int *temp;
	for(int i = 0;i < k; i++){
		isingOnce(G, R, w, n);

    // temp = G;
    // G = R;
    // R = temp;
		 for(int i =0;i< n;i++){
			for(int j = 0;j < n;j++){
				G[i*n+j] = R[i*n+j];
			}
		}
	}
  free(R);
}

// This function executes only one iteration of the ising model
// It loops through all the points and writes the updated spins in an intermidiate matrix.
void isingOnce(int *G, int *R, double *w, int n){
	for(int i = 0; i < n; i++){
		for( int j = 0; j < n; j++){
			int pos = i * n + j;
			R[pos] = updateSpin(G, w, i, j, n);
		}
	}
}

//This function updates the spin of a single moment.
//The Grid Matrix G is indexed correctly with the use of posRoll.
//The functions returns the updated spin value.

int updateSpin(int *G, double *w, int x, int y, int n){
	double sum = 0.0;
	for(int i = 0; i < 5; i++){
		for(int j = 0; j < 5; j++){
			sum = sum + w[(i * 5 + j)] * G[posRoll((x+i-2),(y-j+2),n)];
		}
	}

	if(fabs(sum) < 10e-5){
		return G[x * n + y];
	}else if(sum > 0){
		return 1;
	}else{
		return -1;
	}

}

//This function rolls 2D coordinates around the matrix
//and returns the corresponding 1D index.
int posRoll(int i, int j, int n){
	 int l = (i >= 0)?(i%n):((i+n)%n);
	 int k = (j >= 0)?(j%n):((j+n)%n);

	return (l * n + k);
}
