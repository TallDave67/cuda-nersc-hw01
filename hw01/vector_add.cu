#include <stdio.h>
#include <random>

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


const int DSIZE = 1024;
const int block_size = 256;  // CUDA maximum is 1024

// vector add kernel: C = A + B
__global__ void vadd(const float* A, const float* B, float* C, int ds) {

    // create typical 1D thread index from built-in variables
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < ds)
    {
        // do the vector (element) add here
        C[idx] = A[idx] + B[idx];
    }
}

int main_vector_add() {

    // *******************************************
    // ********** ALLOCATE & INITIALIZE **********
    // *************** HOST MEMORY ***************
    // *******************************************

    float* h_A, * h_B, * h_C, * d_A, * d_B, * d_C;
    h_A = new float[DSIZE];  // allocate space for vectors in host memory
    h_B = new float[DSIZE];
    h_C = new float[DSIZE];

    // Random seed
    std::random_device rd;

    // Initialize Mersenne Twister pseudo-random number generator
    std::mt19937 gen(rd());

    // Generate pseudo-random numbers
    // uniformly distributed in range (1, 2*DSIZE)
    std::uniform_int_distribution<> dis(1, 2*DSIZE);

    for (int i = 0; i < DSIZE; i++) {  // initialize vectors in host memory
        h_A[i] = static_cast<float>(dis(gen)) / static_cast<float>(2*DSIZE);
        h_B[i] = static_cast<float>(dis(gen)) / static_cast<float>(2*DSIZE);
        h_C[i] = 0;
    }

    // *******************************************
    // ***************** ALLOCATE ****************
    // ************** DEVICE MEMORY **************
    // *******************************************

    cudaMalloc(&d_A, DSIZE * sizeof(float));  // allocate device space for vector A
    cudaMalloc(&d_B, DSIZE * sizeof(float)); // allocate device space for vector B
    cudaMalloc(&d_C, DSIZE * sizeof(float)); // allocate device space for vector C
    cudaCheckErrors("cudaMalloc failure"); // error checking

    // *******************************************
    // ****************** STEP 1 *****************
    // ************* COPY HOST MEMORY ************
    // ************* TO DEVICE MEMORY ************
    // *******************************************

    // copy vector A to device:
    cudaMemcpy(d_A, h_A, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    // copy vector B to device:
    cudaMemcpy(d_B, h_B, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");
    //cuda processing sequence step 1 is complete

    // *******************************************
    // ****************** STEP 2 *****************
    // ****************** EXECUTE ****************
    // **************** DEVICE CODE **************
    // *******************************************

    vadd << <(DSIZE + block_size - 1) / block_size, block_size >> > (d_A, d_B, d_C, DSIZE);
    cudaCheckErrors("kernel launch failure");
    //cuda processing sequence step 2 is complete
    
    // *******************************************
    // ****************** STEP 3 *****************
    // ************ COPY DEVICE MEMORY ***********
    // ************** TO HOST MEMORY *************
    // *******************************************

    // copy vector C from device to host:
    cudaMemcpy(h_C, d_C, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
    //cuda processing sequence step 3 is complete
        
    // *******************************************
    // ************** REPORT RESULTS *************
    // *******************************************

    for (int i = 0; i < DSIZE; i++)
    {
        printf("A[%d] = %f, ", i, h_A[i]);
        printf("B[%d] = %f, ", i, h_B[i]);
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // *******************************************
    // ******************* FREE ******************
    // *************** HOST MEMORY ***************
    // *******************************************

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    // *******************************************
    // ******************* FREE ******************
    // ************** DEVICE MEMORY **************
    // *******************************************

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

