#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define N 1000000  

__global__ void vectorMultiply(float *a, float *b, float *c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

void vectorMultiplyCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] * b[i];
    }
}

int main() {
    int size = N * sizeof(float);

    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c_cpu = (float *)malloc(size);
    float *h_c_gpu = (float *)malloc(size);

    for (int i = 0; i < N; i++) {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }

    clock_t start_cpu = clock();
    vectorMultiplyCPU(h_a, h_b, h_c_cpu, N);
    clock_t end_cpu = clock();
    double cpu_time = 1000.0 * (end_cpu - start_cpu) / CLOCKS_PER_SEC;
    printf("CPU Time: %.3f ms\n", cpu_time);

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);
    vectorMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU Time: %.3f ms\n", gpu_time);

    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);

    int correct = 1;
    for (int i = 0; i < N; i++) {
        if (abs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
            correct = 0;
            break;
        }
    }
    if (correct) {
        printf("Matching results\n");
    } else {
        printf("Results Mismatched\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
