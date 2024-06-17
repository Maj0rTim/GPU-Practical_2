// Timothy Fischer

#include <iostream>

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include <helper_cuda.h>

#define N 1024
#define SIZE 1000
#define MAX_ITER 100
#define MAX_TEMP_ERROR 0.01

double Temperature[N][N];
double Temperature_last[N][N];

double* d_Temperature;
double* d_Temperature_last;

void init()
{
    int i,j;

    // set cnter grid 
    for(i = 0; i <= SIZE+1; i++){
        for (j = 0; j <= SIZE+1; j++){
            Temperature_last[i+1][j+1] = 0.0;
        }
    }

    // set left side to 0 and right to a linear increase
    for(i = 0; i <= SIZE+1; i++) {
        Temperature_last[i][0] = 0.0;
        Temperature_last[i][SIZE+1] = (100.0/SIZE)*i;
    }
    
    // set top to 0 and bottom to linear increase
    for(j = 0; j <= SIZE+1; j++) {
        Temperature_last[0][j] = 0.0;
        Temperature_last[SIZE+1][j] = (100.0/SIZE)*j;
    }
}

__device__ double my_fmax(double a, double b) 
{
  return (a > b) ? a : b;
}

__global__ void updateTemp(double** Temperature, double** Temperature_last, int max_iter, double max_error)
{
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);

    int iter = 0;
    int dt = 100;

    int i = idx%1024;
    int j = (int)idx/1024;

    while(iter < max_iter && dt > max_error)
    {
        if (i < SIZE+2 && j < SIZE+2)
        {
            Temperature[i][j] = 0.25 * (Temperature_last[i+1][j] + Temperature_last[i-1][j] + Temperature_last[i][j+1] + Temperature_last[i][j-1]);

            iter++;

            dt = my_fmax( fabs(Temperature[i][j]-Temperature_last[i][j]), dt);
            Temperature_last[i][j] = Temperature[i][j];
        }
    }
}


int main(int argc, char** argv)
{ 
    init();

    // initialise CUDA timing
	float milli;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    int max_iter = MAX_ITER;
    double max_error = MAX_TEMP_ERROR;
    int size = N * N * sizeof(double);

    // Allocate device memory
    checkCudaErrors(cudaMalloc((void**)&d_Temperature, size));
    checkCudaErrors(cudaMalloc((void**)&d_Temperature_last, size));

    // Copy initialized matrix from host to device
    checkCudaErrors(cudaMemcpy(d_Temperature, Temperature, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Temperature_last, Temperature_last, size, cudaMemcpyHostToDevice));

    // Set Kernel Parameters
    dim3 matrixBlock(32, 32, 1);
    dim3 matrixGrid(32, 32, 1);

    // Launch Kernel
    cudaEventRecord(start); 
    checkCudaErrors(cudaDeviceSynchronize());
    updateTemp <<< matrixBlock, matrixGrid >>> (&d_Temperature, &d_Temperature_last, max_iter, max_error);
    cudaEventRecord(stop);
	checkCudaErrors(cudaEventSynchronize(stop));
	cudaEventElapsedTime(&milli, start, stop);  

    printf("updateTemp <<<(%d,%d), (%d,%d)>>> (ms): %f \n", matrixGrid.x, matrixGrid.y,
        matrixBlock.x, matrixBlock.y, milli);

    checkCudaErrors(cudaFree(Temperature));
    checkCudaErrors(cudaFree(Temperature_last));
    


    return 0;
}
