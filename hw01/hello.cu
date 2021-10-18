#include <stdio.h>

__global__ void hello() {

    printf("Hello from block: %u, thread: %u\n", threadIdx.x, blockIdx.x);
}

int main_hello() {

    hello << <16, 16 >> > ();
    cudaDeviceSynchronize();
    return 0;
}

