#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <chrono>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <string>
#include <vector>


#define MOMENTS_PER__BLOCK (107)
#define MOMENTS_PER_THREAD 5
#define ZERO_THRESHOLD 1e-6
#define WEIGHT_SIZE 5
#define SPIN_SIZE 517

void fillLine(int * Mask, int n, sf::Vector2i p, int range, int fill_value);
sf::VertexArray createGrid(int num_of_cells, int dim);
int posRollHost(int i, int j, int n);
void drawGrid(int num_of_cells, sf::VertexArray & grid, int dim, int * model);
__device__ int updateSpin(int* G, float* w, int x, int y, int n);
__global__ void isingOnce(int* G, int* R, float* w, int n);
__device__ int posRoll(int i, int j, int n);
__device__ void readElements(int *g_mem, int *s_mem, int g_index, int s_index, int elements);
void readFile(const char *file_name, int *Table, int n);
template <class T> void swap(T **x, T **y);
bool validation(int *Result, int *Solution, int n);


static const char *RESULT_STATUS[] = { "WRONG", "CORRECT" };

const float M[(WEIGHT_SIZE * WEIGHT_SIZE)] = {
	0.004,  0.016,  0.026,  0.016,   0.004,
	0.016,  0.071,  0.117,  0.071,   0.016,
	0.026,  0.117,  0.00,    0.117,   0.026,
	0.016,  0.071, 0.117,  0.071,   0.016,
	0.004,  0.016,  0.026,  0.016,   0.004 };


int main()
{
	std::string file_name = "";
	int *r_A = (int *)malloc(SPIN_SIZE*SPIN_SIZE * sizeof(int));
	int *A = (int *)malloc(SPIN_SIZE*SPIN_SIZE * sizeof(int));
	int *MODEL_MASK = (int *)malloc(SPIN_SIZE*SPIN_SIZE * sizeof(int));


	for (int i = 0;i < SPIN_SIZE;i++) {
		for (int j = 0; j < SPIN_SIZE;j++) {
			A[i*SPIN_SIZE + j] = 0;
			MODEL_MASK[i*SPIN_SIZE + j] = 0;
		}
	}

	bool has_changed = false;
	bool start_recording = false;
	bool start = false;
	int *d_A, *d_R;
	float* W;
	int n = 517;
	int k = 1;

	size_t bytes_of_model = n * n * sizeof(int);
	size_t bytes_of_weight = (WEIGHT_SIZE * WEIGHT_SIZE) * sizeof(float);

	cudaMalloc(&d_A, bytes_of_model);
	cudaMalloc(&d_R, bytes_of_model);
	cudaMalloc(&W, bytes_of_weight);


	dim3 dimBlock(1 << 3, 1 << 5);
	dim3 dimGrid(517, 1);
	int shared_mem = (517 + 4) * 5 * sizeof(int);

	int framerate = 60;
	sf::Vector2f viewSize(517, 517);
	sf::VideoMode vm(viewSize.x, viewSize.y);
	sf::RenderWindow window(vm, "Evolution of Ising Model", sf::Style::Close);


	int fill_value = 1;
	int millis = 50;
	sf::Clock clock;
	sf::Time elapsed_time = clock.getElapsedTime();
	sf::Time delta_time = sf::milliseconds(millis);


	sf::VertexArray grid = createGrid(517, 1);


	while (window.isOpen())
	{
		sf::Event event;
		while (window.pollEvent(event))
		{
			switch (event.type)
			{
			case sf::Event::Closed:
				window.close();
				break;
			case sf::Event::KeyPressed:
				if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) {
					window.close();
				}
				else if (sf::Keyboard::isKeyPressed(sf::Keyboard::I)) {
					FILE *fptr = fopen("conf-init.bin", "r");
					if (fptr == NULL) {
						printf("Could not open this file!\n");
					}
					else {
						for (int i = 0;i < SPIN_SIZE; i++)
							for (int j = 0;j < SPIN_SIZE; j++)
								fread(&r_A[i * SPIN_SIZE + j], sizeof(int), 1, fptr);
						has_changed = true;
						swap(&A, &r_A);
						printf("Imported Succesfully!\n");
					}
				}
				else if (sf::Keyboard::isKeyPressed(sf::Keyboard::B)) {
					start = true;
				}
				else if (sf::Keyboard::isKeyPressed(sf::Keyboard::F)) {
					start = false;
				}
				break;
			case sf::Event::MouseButtonPressed:
				if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
					std::cout << "start recording\n";
					start_recording = true;
					fill_value = 1;
				}
				else if (sf::Mouse::isButtonPressed(sf::Mouse::Right)) {
					std::cout << "start recording\n";
					start_recording = true;
					fill_value = -1;
				}
				break;
			case sf::Event::MouseButtonReleased:
				std::cout << "stop recording\n";
				start_recording = false;


				break;
			case sf::Event::MouseMoved:
				if (start_recording) {
					fillLine(A, n, sf::Mouse::getPosition(window), 25, fill_value);
				}
				has_changed = true;
				break;
			default:
				break;
			}
		}

		if (has_changed) {
			has_changed = false;
			//grid = createGrid(517, 1, A);
			drawGrid(517, grid, 1, A);
		}

		if ((clock.getElapsedTime() - elapsed_time) > delta_time && start && !start_recording) {
			elapsed_time = clock.getElapsedTime();


			delta_time = sf::milliseconds(millis);
			//std::cout << "Evolving...\n";

			cudaMemcpy(d_A, A, bytes_of_model, cudaMemcpyHostToDevice);
			cudaMemcpy(W, M, bytes_of_weight, cudaMemcpyHostToDevice);

			isingOnce << <dimGrid, dimBlock, shared_mem >> > (d_A, d_R, W, n);

			swap(&d_A, &d_R);

			cudaMemcpy(A, d_A, bytes_of_model, cudaMemcpyDeviceToHost);
			has_changed = true;
		}




		//window.clear(sf::Color::White);
		window.draw(grid);
		window.display();
	}

	cudaFree(d_A);
	cudaFree(d_R);
	cudaFree(W);
	free(A);


	return EXIT_SUCCESS;
}




sf::VertexArray createGrid(int num_of_cells, int dim) {
	int pos_x = 0;
	int pos_y = 0;

	int num_of_coordinates = 4 * (num_of_cells * num_of_cells);

	sf::VertexArray grid(sf::Quads, num_of_coordinates);
	for (int j = 0; j < num_of_cells;j++) {
		for (int i = 0; i < num_of_cells;i++) {
			grid[4 * i + j * 4 * num_of_cells].position = sf::Vector2f(pos_x + i * dim, pos_y + j * dim);
			grid[4 * i + 1 + j * 4 * num_of_cells].position = sf::Vector2f(pos_x + i * dim, pos_y + dim + j * dim);
			grid[4 * i + 2 + j * 4 * num_of_cells].position = sf::Vector2f(pos_x + dim + i * dim, pos_y + dim + j * dim);
			grid[4 * i + 3 + j * 4 * num_of_cells].position = sf::Vector2f(pos_x + dim + i * dim, pos_y + j * dim);

		}
	}


	return grid;

}




void drawGrid(int num_of_cells, sf::VertexArray & grid, int dim, int * model)
{
	for (int j = 0; j < num_of_cells;j++) {
		for (int i = 0; i < num_of_cells;i++) {
			if (model[j*num_of_cells + i] == -1) {
				grid[4 * i + j * 4 * num_of_cells].color = sf::Color::Green;
				grid[4 * i + 1 + j * 4 * num_of_cells].color = sf::Color::Green;
				grid[4 * i + 2 + j * 4 * num_of_cells].color = sf::Color::Green;
				grid[4 * i + 3 + j * 4 * num_of_cells].color = sf::Color::Green;
			}
			else {
				grid[4 * i + j * 4 * num_of_cells].color = sf::Color::Red;
				grid[4 * i + 1 + j * 4 * num_of_cells].color = sf::Color::Red;
				grid[4 * i + 2 + j * 4 * num_of_cells].color = sf::Color::Red;
				grid[4 * i + 3 + j * 4 * num_of_cells].color = sf::Color::Red;
			}
		}
	}


}



void fillLine(int * Mask, int n, sf::Vector2i p, int range, int fill_value) {
	int begin = -range / 2;
	int end = range - range / 2;


	for (int i = begin; i < end;i++) {
		for (int j = begin;j < end; j++) {
			if (0 <= i && i < 517 && 0 <= j && j < 517)
				Mask[posRollHost((i + p.y), (j + p.x), n)] = fill_value;
		}
	}
}



int posRollHost(int i, int j, int n) {
	//Roll coordinates
	int l = (i + n) % n;
	int k = (j + n) % n;

	return (l * n + k);
}








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





__device__ int updateSpin(int* G, float* w, int x, int y, int n) {
	float sum = 0.0;

	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
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

__device__ int posRoll(int i, int j, int n) {
	//Roll coordinates
	int l = (i + n) % n;
	int k = (j + n) % n;

	return (l * n + k);
}


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


bool validation(int * Result, int * Solution, int n)
{
	int result = 1;
	for (int i = 0;i < n;i++) {
		for (int j = 0;j < n;j++) {
			if (fabs(Result[i * n + j] - Solution[i * n + j]) > 1) {
				std::cout << i << ", " << j << " " << Result[i * n + j] << " " << Solution[i * n + j] << std::endl;
				result = 0;
				return result;
			}
		}
	}
	return result;
}

template <class T>
void swap(T **x, T **y) {
	T *temp = *x;
	*x = *y;
	*y = temp;
}

