#include <iostream>
#include <cuda_runtime.h>
using namespace std;
__global__ void helloWorldKernel()
{
    printf("Hello from GPU! Thread ID: %d\n", threadIdx.x);
}

int main()
{
    helloWorldKernel<<<1, 10>>>();

    cudaDeviceSynchronize();

    cout << "Hello from CPU!" << endl;

    return 0;
}