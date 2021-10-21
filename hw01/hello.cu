#include <stdio.h>
#include <stdlib.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

#define TWO_LESS_THAN_TWO_TO_THE_TWENTYFOUR 16777214.0f
#define SMALL_NUMBER 7.0f

__global__ void hello(float* f, double* d) {

    printf("Hello from block: %u, thread: %u\n", threadIdx.x, blockIdx.x);
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        f[0] += SMALL_NUMBER;
        d[0] += SMALL_NUMBER;
    }
}


int main_hello() {

    // *******************************************
    // ********** ALLOCATE & INITIALIZE **********
    // *************** HOST MEMORY ***************
    // *******************************************

    float* h_f = nullptr;
    double* h_d = nullptr;
    h_f = new float[1];
    h_d = new double[1];
    h_f[0] = TWO_LESS_THAN_TWO_TO_THE_TWENTYFOUR;
    h_d[0] = TWO_LESS_THAN_TWO_TO_THE_TWENTYFOUR;

    // 
    // *******************************************
    // ***************** ALLOCATE ****************
    // ************** DEVICE MEMORY **************
    // *******************************************

    float * d_f = nullptr;
    double * d_d = nullptr;
    cudaMalloc(&d_f, sizeof(float));
    cudaMalloc(&d_d, sizeof(double));
    cudaCheckErrors("cudaMalloc failure");

    // *******************************************
    // ****************** STEP 1 *****************
    // ************* COPY HOST MEMORY ************
    // ************* TO DEVICE MEMORY ************
    // *******************************************

    //copy input data over to GPU
    cudaMemcpy(d_f, h_f, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d, sizeof(double), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");
    // Cuda processing sequence step 1 is complete

    // *******************************************
    // ****************** STEP 2 *****************
    // ****************** EXECUTE ****************
    // **************** DEVICE CODE **************
    // *******************************************

    // Launch kernel
    hello <<<2, 2>>>(d_f, d_d);
    cudaCheckErrors("kernel launch failure");
    // Cuda processing sequence step 2 is complete

    // *******************************************
    // ****************** STEP 3 *****************
    // ************ COPY DEVICE MEMORY ***********
    // ************** TO HOST MEMORY *************
    // *******************************************

    // Copy results back to host
    cudaMemcpy(h_f, d_f, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d, d_d, sizeof(double), cudaMemcpyDeviceToHost);
    // Cuda processing sequence step 3 is complete

    // *******************************************
    // **************** VERIFY AND ***************
    // ************** REPORT RESULTS *************
    // *******************************************

    // Report results
    printf("current float: %.8f, current double: %.8f\n", h_f[0], h_d[0]);

    // Comare resulst
    float ho_f = TWO_LESS_THAN_TWO_TO_THE_TWENTYFOUR;
    double ho_d = TWO_LESS_THAN_TWO_TO_THE_TWENTYFOUR;
    ho_f += SMALL_NUMBER;
    ho_d += SMALL_NUMBER;
    printf("host-only float: %.8f, host-only double: %.8f\n", ho_f, ho_d);

    // *******************************************
    // ******************* FREE ******************
    // *************** HOST MEMORY ***************
    // *******************************************

    delete [] h_f;
    delete [] h_d;

    // *******************************************
    // ******************* FREE ******************
    // ************** DEVICE MEMORY **************
    // *******************************************

    cudaFree(d_f);
    cudaFree(d_d);

    return 0;
}

