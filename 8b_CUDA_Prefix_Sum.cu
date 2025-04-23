#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>

#define N 10
#define BLOCKSIZE 1024

__global__ void PrefixSum(float *dInArray, float *dOutArray, int arrayLen, int threadDim) {
    int tid = threadIdx.x;
    __shared__ float temp[BLOCKSIZE];

    if (tid < arrayLen) {
        temp[tid] = dInArray[tid];
    }

    __syncthreads();

    for (int offset = 1; offset < threadDim; offset *= 2) {
        float val = 0;
        if (tid >= offset)
            val = temp[tid - offset];

        __syncthreads();

        if (tid < arrayLen)
            temp[tid] += val;

        __syncthreads();
    }

    if (tid < arrayLen) {
        dOutArray[tid] = temp[tid];
    }
}

int main() {
    float x_h[N], y_h[N];
    float *x_d, *y_d;

    for (int i = 0; i < N; i++)
        x_h[i] = i;

    cudaMalloc((void**)&x_d, N * sizeof(float));
    cudaMalloc((void**)&y_d, N * sizeof(float));

    cudaMemcpy(x_d, x_h, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCKSIZE);
    dim3 grid(1);
    PrefixSum<<<grid, block>>>(x_d, y_d, N, BLOCKSIZE);
    cudaDeviceSynchronize();

    cudaMemcpy(y_h, y_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Input Array:");
    for (int i = 0; i < N; i++) printf("%.0f ", x_h[i]);
    printf("\nPrefix Array:");
    for (int i = 0; i < N; i++) printf("%.0f ", y_h[i]);
    printf("\n");

    cudaFree(x_d);
    cudaFree(y_d);

    return 0;
}
