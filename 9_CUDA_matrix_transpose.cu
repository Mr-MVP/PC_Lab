#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024  

__global__ void transposeGPU(float *d_out, float *d_in, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < width) {
        int in_idx = y * width + x;
        int out_idx = x * width + y;
        d_out[out_idx] = d_in[in_idx];
    }
}

void transposeCPU(float *out, float *in, int width) {
    for (int y = 0; y < width; y++) {
        for (int x = 0; x < width; x++) {
            out[x * width + y] = in[y * width + x];
        }
    }
}

int main() {
    int size = N * N * sizeof(float);

    float *h_in = (float*)malloc(size);
    float *h_out_cpu = (float*)malloc(size);
    float *h_out_gpu = (float*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_in[i] = rand() % 100;
    }

    clock_t start_cpu = clock();
    transposeCPU(h_out_cpu, h_in, N);
    clock_t end_cpu = clock();
    double cpu_time = 1000.0 * (end_cpu - start_cpu) / CLOCKS_PER_SEC;
    printf("CPU Time: %.3f ms\n", cpu_time);

    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    transposeGPU<<<blocks, threads>>>(d_out, d_in, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU Time: %.3f ms\n", gpu_time);

    cudaMemcpy(h_out_gpu, d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out_cpu);
    free(h_out_gpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}