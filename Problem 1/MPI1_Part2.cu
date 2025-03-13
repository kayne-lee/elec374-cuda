/* Kayne Lee, 20350003 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>

#define NUM_SIZES 5

// different matrix sizes to test
int sizes[NUM_SIZES] = { 256, 512, 1024, 2048, 4096 };

// function to measure memory transfer time (part 1 of part 2)
void measureTransferTime(int n) {
    size_t bytes = n * n * sizeof(float);

    // allocate host memory
    float* h_matrix = (float*)malloc(bytes);
    for (int i = 0; i < n * n; i++) {
        h_matrix[i] = (float)(rand() % 100);
    }

    // allocate device memory
    float* d_matrix;
    cudaMalloc((void**)&d_matrix, bytes);

    // create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float timeH2D = 0.0f, timeD2H = 0.0f;

    // measure Host to Device (H2D) transfer time
    cudaEventRecord(start, 0);
    cudaMemcpyAsync(d_matrix, h_matrix, bytes, cudaMemcpyHostToDevice, 0);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeH2D, start, stop);

    // measure Device to Host (D2H) transfer time
    cudaEventRecord(start, 0);
    cudaMemcpyAsync(h_matrix, d_matrix, bytes, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeD2H, start, stop);

    // print results
    printf("Matrix Size: %d x %d | H2D Transfer: %.3f ms | D2H Transfer: %.3f ms\n", n, n, timeH2D, timeD2H);

    // cleanup
    free(h_matrix);
    cudaFree(d_matrix);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Function to multiply matrices on the CPU
void multiplyMatricesCPU(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// CUDA kernel for matrix multiplication (single block, one thread)
__global__ void multiplyMatricesGPU(float* A, float* B, float* C, int n) {
    int i = 0, j = 0;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// Function to measure CPU matrix multiplication time
void measureMatrixMultiplicationCPU(int n) {
    size_t bytes = n * n * sizeof(float);
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

    for (int i = 0; i < n * n; i++) {
        h_A[i] = (float)(rand() % 100);
        h_B[i] = (float)(rand() % 100);
    }

    clock_t start = clock();
    multiplyMatricesCPU(h_A, h_B, h_C, n);
    clock_t end = clock();

    printf("CPU Matrix Multiplication Time for %d x %d: %.6f seconds\n", n, n, (double)(end - start) / CLOCKS_PER_SEC);

    free(h_A);
    free(h_B);
    free(h_C);
}

// Function to measure GPU matrix multiplication time (including transfer time)
void measureMatrixMultiplicationGPU(int n) {
    size_t bytes = n * n * sizeof(float);
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);
    float* d_A, * d_B, * d_C;

    for (int i = 0; i < n * n; i++) {
        h_A[i] = (float)(rand() % 100);
        h_B[i] = (float)(rand() % 100);
    }

    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    // Measure H2D transfer time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float timeH2D = 0.0f;
    cudaEventRecord(start, 0);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeH2D, start, stop);

    // Launch the GPU kernel
    cudaEventRecord(start, 0);
    multiplyMatricesGPU <<<1, 1 >>> (d_A, d_B, d_C, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float kernelTime;
    cudaEventElapsedTime(&kernelTime, start, stop);

    // Measure D2H transfer time
    float timeD2H = 0.0f;
    cudaEventRecord(start, 0);
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeD2H, start, stop);

    printf("GPU Matrix Multiplication Time for %d x %d: %.6f seconds (kernel: %.6f + H2D: %.3f + D2H: %.3f)\n",
        n, n, (kernelTime + timeH2D + timeD2H) / 1000.0f, kernelTime / 1000.0f, timeH2D, timeD2H);

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    srand(time(NULL));

    printf("Measuring Memory Transfer Time (Host ↔ Device)\n");
    printf("---------------------------------------------------\n");

    for (int i = 0; i < NUM_SIZES; i++) {
        measureTransferTime(sizes[i]);
    }

    printf("\nMatrix Multiplication CPU vs GPU\n");
    printf("---------------------------------\n");
    for (int i = 0; i < NUM_SIZES; i++) {
        int size = sizes[i];
        measureMatrixMultiplicationCPU(size);
        measureMatrixMultiplicationGPU(size);
    }

    return 0;
}
