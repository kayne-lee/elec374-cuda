/* kayne lee, 20350003*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#define TILE_H 12
#define TILE_W 18

void init_rand_mat(float* mat, int rows, int cols) {
    // fill matrix with random floats 1-255
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i*cols + j] = 1 + (float)rand()/(float)RAND_MAX * 254.0f;
        }
    }
}

void cpu_matmul(const float* a, const float* b, float* res, int m, int k, int n) {
    // compute matrix product on host
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int l = 0; l < k; l++) {
                sum += a[i*k + l] * b[l*n + j];
            }
            res[i*n + j] = sum;
        }
    }
}

int verify_result(float* a, float* b, float* gpu_res, int m, int k, int n) {
    // validate gpu result against cpu computation
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int l = 0; l < k; l++) {
                sum += a[i*k + l] * b[l*n + j];
            }
            if (fabs(sum - gpu_res[i*n + j]) > 1e-3) return 0;
        }
    }
    return 1;
}

__global__ void gpu_mmul_tiled(const float* a, const float* b, float* res, 
                              int m, int k, int n) {
    __shared__ float s_a[TILE_H][TILE_H];
    __shared__ float s_b[TILE_H][TILE_H];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    
    int row = by * TILE_H + ty;
    int col = bx * TILE_W + tx;
    
    float acc = 0.0f;
    
    for (int phase = 0; phase < (k + TILE_H - 1)/TILE_H; phase++) {
        int a_col = phase * TILE_H + tx;
        int b_row = phase * TILE_H + ty;
        
        s_a[ty][tx] = (row < m && a_col < k) ? a[row*k + a_col] : 0;
        s_b[ty][tx] = (b_row < k && col < n) ? b[b_row*n + col] : 0;
        
        __syncthreads();
        
        for (int l = 0; l < TILE_H; l++) {
            acc += s_a[ty][l] * s_b[l][tx];
        }
        __syncthreads();
    }
    
    if (row < m && col < n) {
        res[row*n + col] = acc;
    }
}

void get_kernel_stats() {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, gpu_mmul_tiled);
    
    int block_size = TILE_W * TILE_H;
    int max_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, gpu_mmul_tiled, block_size, 0);
    
    printf("[kernel stats]\n");
    printf("tile: %dx%d | regs/thread: %d | shared/block: %zu bytes\n", 
           TILE_H, TILE_W, attr.numRegs, attr.sharedSizeBytes);
    printf("max blocks per sm: %d | max threads per sm: %d\n\n", 
           max_blocks, max_blocks * block_size);
}

void run_test(int m, int k, int n) {
    printf("\n>> testing %dx%d * %dx%d\n", m, k, k, n);
    
    float *host_a, *host_b, *host_res;
    host_a = (float*)malloc(m*k*sizeof(float));
    host_b = (float*)malloc(k*n*sizeof(float));
    host_res = (float*)malloc(m*n*sizeof(float));
    
    init_rand_mat(host_a, m, k);
    init_rand_mat(host_b, k, n);
    
    // gpu setup
    float *dev_a, *dev_b, *dev_res;
    cudaMalloc(&dev_a, m*k*sizeof(float));
    cudaMalloc(&dev_b, k*n*sizeof(float));
    cudaMalloc(&dev_res, m*n*sizeof(float));
    
    cudaMemcpy(dev_a, host_a, m*k*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, k*n*sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 block(TILE_W, TILE_H);
    dim3 grid((n + TILE_W-1)/TILE_W, (m + TILE_H-1)/TILE_H);
    
    // gpu timing
    cudaEvent_t g_start, g_stop;
    cudaEventCreate(&g_start);
    cudaEventCreate(&g_stop);
    cudaEventRecord(g_start);
    
    gpu_mmul_tiled<<<grid, block>>>(dev_a, dev_b, dev_res, m, k, n);
    
    cudaEventRecord(g_stop);
    cudaEventSynchronize(g_stop);
    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, g_start, g_stop);
    
    cudaMemcpy(host_res, dev_res, m*n*sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("gpu time: %.2f ms | ", gpu_ms);
    printf("verify: %s\n", verify_result(host_a, host_b, host_res, m, k, n) ? "passed" : "failed");
    
    free(host_a); free(host_b); free(host_res);
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_res);
}

int main() {
    srand(time(0));
    get_kernel_stats();
    
    // Bonus mark test cases
    run_test(750, 800, 850);
    run_test(2000, 1750, 1900);
    
    return 0;
}
