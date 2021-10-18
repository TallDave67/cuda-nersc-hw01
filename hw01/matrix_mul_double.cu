#include <stdio.h>

// these are just for timing measurments
#include <time.h>

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
const int block_size = 16;  // CUDA maximum is 1024 *total* threads in block
const double B_val = 2.0f;

// matrix multiply (naive) kernel: C = A * B
__global__ void mmul_double(const double* A, const double* B, double* C, int ds) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x; // create thread x index
    int idy = threadIdx.y + blockDim.y * blockIdx.y; // create thread y index

    if ((idx < ds) && (idy < ds)) {
        double temp = 0;
        for (int i = 0; i < ds; i++)
            temp += A[idy * ds + i] * B[i * ds + idx];   // dot product of row and column
        C[idy * ds + idx] = temp;
    }
}

int main_matrix_mul_double() {

    // *******************************************
    // ********** ALLOCATE & INITIALIZE **********
    // *************** HOST MEMORY ***************
    // *******************************************

    double* h_A, * h_B, * h_C, * d_A, * d_B, * d_C;

    h_A = new double[DSIZE * DSIZE];
    h_B = new double[DSIZE * DSIZE];
    h_C = new double[DSIZE * DSIZE];
    for (int i = 0; i < DSIZE * DSIZE; i++) {
        h_A[i] = static_cast<double>(i);
        h_B[i] = B_val;
        h_C[i] = 0;
    }

    // *******************************************
    // ***************** ALLOCATE ****************
    // ************** DEVICE MEMORY **************
    // *******************************************

    // these are just for timing
    clock_t t0, t1, t2;
    double t1sum = 0.0;
    double t2sum = 0.0;

    // start timing
    t0 = clock();

    // Initialization timing
    t1 = clock();
    t1sum = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);

    // Allocate device memory
    cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(double));
    cudaMalloc(&d_B, DSIZE * DSIZE * sizeof(double));
    cudaMalloc(&d_C, DSIZE * DSIZE * sizeof(double));
    cudaCheckErrors("cudaMalloc failure");

    // *******************************************
    // ****************** STEP 1 *****************
    // ************* COPY HOST MEMORY ************
    // ************* TO DEVICE MEMORY ************
    // *******************************************

    //copy input data over to GPU
    cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * DSIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");
    // Cuda processing sequence step 1 is complete

    // *******************************************
    // ****************** STEP 2 *****************
    // ****************** EXECUTE ****************
    // **************** DEVICE CODE **************
    // *******************************************

    // Launch kernel
    dim3 block(block_size, block_size);  // dim3 variable holds 3 dimensions
    int gx = (DSIZE + block.x - 1) / block.x;
    int gy = (DSIZE + block.y - 1) / block.y;
    printf("gx: %d, gy: %d\n", gx, gy);
    dim3 grid(gx, gy);
    mmul_double << <grid, block >> > (d_A, d_B, d_C, DSIZE);
    cudaCheckErrors("kernel launch failure");
    // Cuda processing sequence step 2 is complete

    // *******************************************
    // ****************** STEP 3 *****************
    // ************ COPY DEVICE MEMORY ***********
    // ************** TO HOST MEMORY *************
    // *******************************************

    // Copy results back to host
    cudaMemcpy(h_C, d_C, DSIZE * DSIZE * sizeof(double), cudaMemcpyDeviceToHost);

    // GPU timing
    t2 = clock();
    t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
    printf("Done. Compute took %f seconds\n", t2sum);
    // Cuda processing sequence step 3 is complete

    // *******************************************
    // **************** VERIFY AND ***************
    // ************** REPORT RESULTS *************
    // *******************************************

    // Verify results
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
    int last_b = -1;
    printf("verifying results by block: ");
    for (int i = 0; i < DSIZE * DSIZE; i++)
    {
        int b = (i - (i % DSIZE)) / DSIZE; //current block
        int n = ((b + 1) * DSIZE) - 1; //max value in current block
        int spread = n - b * DSIZE;
        double avg = static_cast<double>(b * DSIZE) + static_cast<double>(spread) / 2.0f;
        double sA = DSIZE * avg;
        double s = sA * B_val;
        if (b != last_b)
        {
            last_b = b;
            if (b != 0)
            {
                printf(",");
            }
            printf("%d", b);
        }
        if (h_C[i] != s || i == 16384)
        {
            if (h_C[i] != s)
            {
                printf("\nmismatch at index %d, was: %f, should be: %f\n", i, h_C[i], s);
            }
            else
            {
                printf("\n");
            }

            double sA_corr = 0.0f;
            double firstA = 0.0f;
            double lastA = 0.0f;
            for (int j = b * DSIZE; j <= n; j++)
            {
                if (j == b * DSIZE)
                {
                    firstA = h_A[j];
                }
                else if (j == n)
                {
                    lastA = h_A[j];
                }
                printf("current sum: %.8f, ", sA_corr);
                sA_corr += h_A[j];
                printf("add: %.8f, new sum: %.8f\n", h_A[j], sA_corr); 
            }
            printf("b: %d, n: %d, spread: %d, avg: %f, firstA: %f, lastA: %f, sA: %f, sA_corr: %f\n", b, n, spread, avg, firstA, lastA, sA, sA_corr);
            if (h_C[i] != s)
            {
                return -1;
            }
        }
    }
    printf("\nSuccess!\n");

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

