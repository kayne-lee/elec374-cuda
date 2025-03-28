/* Kayne Lee, 20350003*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void generate_random_matrix(float* matrix, int n) {
    // Generate an NxN matrix filled with random float values
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] = 1 + (float)rand() / ((float)RAND_MAX / (255 - 1));
        }
    }
}

void display_matrix(float* matrix, int n) {
    // Display matrix values
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%6.2f ", matrix[i * n + j]);
        }
        printf("\n");
    }
}

void cpu_matrix_multiplication(float* a, float* b, float* c, int n) {
    // Perform matrix multiplication on the CPU
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            c[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                c[i * n + j] += a[i * n + k] * b[k * n + j]; // Perform dot product of row and column
            }
        }
    }
}

bool check_results(float* a, float* b, float* c, int n) {
    // Verify the correctness of matrix multiplication (true if results are correct)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float temp = 0;
            for (int k = 0; k < n; k++) {
                temp += a[i * n + k] * b[k * n + j]; // Perform dot product of row and column
            }
            if (temp != c[i * n + j]) return false;
        }
    }
    return true;
}

// Kernel for tiled matrix multiplication
__global__ void gpu_tiled_multiply(const float* a, const float* b, float* c, int n, int TILE_SIZE) {
    __shared__ float tile_a[32][32];  // Adjust tile size if necessary
    __shared__ float tile_b[32][32];

    // Calculate thread and block indices
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    // Calculate row and column indices for matrix
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float partial_sum = 0;	// Store partial products for each element

    // Loop through the tiles
    for (int i = 0; i < ceil((float)n / TILE_SIZE); i++) {
        // Load matrix A and B into shared memory
        tile_a[ty][tx] = a[row * n + i * TILE_SIZE + tx];
        tile_b[ty][tx] = b[(i * TILE_SIZE + ty) * n + col];

        __syncthreads(); // Sync threads after loading tiles

        // Perform partial sum for the tile
        for (int i = 0; i < TILE_SIZE; i++)
            partial_sum += tile_a[ty][i] * tile_b[i][tx];

        __syncthreads(); // Sync threads before loading next tile
    }

    // Store result to global memory
    if (row < n && col < n)
        c[row * n + col] = partial_sum;
}

void display_kernel_info(int TILE_SIZE) {
    cudaFuncAttributes attr;
    cudaError_t err = cudaFuncGetAttributes(&attr, gpu_tiled_multiply);

    if (err != cudaSuccess) {
        printf("Error fetching kernel attributes: %s\n", cudaGetErrorString(err));
        return;
    }

    int blockSize = TILE_SIZE * TILE_SIZE;
    int maxActiveBlocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM, gpu_tiled_multiply, blockSize, 0);

    int maxThreadsPerSM = maxActiveBlocksPerSM * blockSize;

    printf("Kernel Info for gpu_tiled_multiply:\n");
    printf("Tile Size: %d x %d\n", TILE_SIZE, TILE_SIZE);
    printf("Thread Registers: %d\n", attr.numRegs);
    printf("Shared Memory per Block: %zu bytes\n", attr.sharedSizeBytes);
    printf("Max Active Blocks per SM: %d\n", maxActiveBlocksPerSM);
    printf("Max Threads per SM: %d\n", maxThreadsPerSM);
    printf("\n");
}

void execute_matrix_multiplication(int n, int TILE_SIZE) {
    printf("Testing matrix size %d x %d with tile size %d... ", n, n, TILE_SIZE);

    // Allocate memory for host matrices
    float* h_matrix_a = (float*)malloc(n * n * sizeof(float));
    float* h_matrix_b = (float*)malloc(n * n * sizeof(float));
    float* h_matrix_c = (float*)malloc(n * n * sizeof(float));
    float* d_matrix_a, * d_matrix_b, * d_matrix_c;

    // Generate random input matrices
    generate_random_matrix(h_matrix_a, n);
    generate_random_matrix(h_matrix_b, n);

    // Allocate memory on device
    cudaMalloc((void**)&d_matrix_a, n * n * sizeof(float));
    cudaMalloc((void**)&d_matrix_b, n * n * sizeof(float));
    cudaMalloc((void**)&d_matrix_c, n * n * sizeof(float));

    // Set up kernel launch configuration
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Create CUDA events for timing
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaDeviceSynchronize(); // Sync GPU before starting

    // Copy data from host to device
    cudaMemcpy(d_matrix_a, h_matrix_a, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_b, h_matrix_b, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Record start time
    cudaEventRecord(start_event, 0);

    // Launch GPU kernel
    gpu_tiled_multiply << <blocksPerGrid, threadsPerBlock >> > (d_matrix_a, d_matrix_b, d_matrix_c, n, TILE_SIZE);

    // Record stop time
    cudaEventRecord(stop_event, 0);

    // Wait for GPU to finish
    cudaEventSynchronize(stop_event);

    // Measure elapsed time
    float gpu_time_ms = 0.0f;
    cudaEventElapsedTime(&gpu_time_ms, start_event, stop_event);

    printf("GPU Time: %.3f ms\n", gpu_time_ms);

    // Copy the result back to host
    cudaMemcpy(h_matrix_c, d_matrix_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);

     // Optionally verify the result
     if (check_results(h_matrix_a, h_matrix_b, h_matrix_c, n)) {
         printf("TEST PASSED\n");
     } else {
         printf("TEST FAILED\n");
     }


    // Clean up resources
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    free(h_matrix_a);
    free(h_matrix_b);
    free(h_matrix_c);
    cudaFree(d_matrix_a);
    cudaFree(d_matrix_b);
    cudaFree(d_matrix_c);
}

int main() {
    srand((unsigned)time(NULL)); // Seed random number generator

    int tile_sizes[] = { 2, 4, 8, 16, 32 }; // Different tile sizes to test

    // Display CUDA kernel info for each tile size
    for (int i = 0; i < 5; i++) {
        display_kernel_info(tile_sizes[i]);
    }

    // Run the matrix multiplication for various matrix sizes and tile sizes
    for (int i = 0; i < 5; i++) {
        execute_matrix_multiplication(256, tile_sizes[i]);
        execute_matrix_multiplication(512, tile_sizes[i]);
        execute_matrix_multiplication(1024, tile_sizes[i]);
        execute_matrix_multiplication(2048, tile_sizes[i]);
        execute_matrix_multiplication(4096, tile_sizes[i]);
    }

    return 0;
}
