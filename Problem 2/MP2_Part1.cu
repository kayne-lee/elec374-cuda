/* kayne lee, 20350003*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void init_rand_matrix(float* m, int dim) {
    // fill matrix with random floats between 1 and 255
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            m[i*dim + j] = 1 + (float)rand() / ((float)RAND_MAX / 254.0f);
        }
    }
}

void print_matrix(float* m, int dim) {
    // display matrix contents
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            printf("%6.2f ", m[i*dim + j]);
        }
        putchar('\n');
    }
}

void cpu_mmul(float* a, float* b, float* res, int n) {
    // compute matrix product on host
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int k = 0; k < n; k++) {
                sum += a[i*n + k] * b[k*n + j];
            }
            res[i*n + j] = sum;
        }
    }
}

int verify_result(float* a, float* b, float* gpu_res, int n) {
    // validate gpu result against cpu computation
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int k = 0; k < n; k++) {
                sum += a[i*n + k] * b[k*n + j];
            }
            if (fabs(sum - gpu_res[i*n + j]) > 1e-3) return 0;
        }
    }
    return 1;
}

__global__ void gpu_mmul_tiled(const float* a, const float* b, float* res, int dim, int tile) {
    __shared__ float s_a[32][32];
    __shared__ float s_b[32][32];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    
    int row = by * tile + ty;
    int col = bx * tile + tx;
    
    float acc = 0.0f;
    
    for (int phase = 0; phase < (dim + tile - 1)/tile; phase++) {
        int a_col = phase * tile + tx;
        int b_row = phase * tile + ty;
        
        s_a[ty][tx] = (row < dim && a_col < dim) ? a[row*dim + a_col] : 0;
        s_b[ty][tx] = (b_row < dim && col < dim) ? b[b_row*dim + col] : 0;
        
        __syncthreads();
        
        for (int k = 0; k < tile; k++) {
            acc += s_a[ty][k] * s_b[k][tx];
        }
        __syncthreads();
    }
    
    if (row < dim && col < dim) {
        res[row*dim + col] = acc;
    }
}

void get_kernel_stats(int tile) {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, gpu_mmul_tiled);
    
    int block_size = tile * tile;
    int max_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, gpu_mmul_tiled, block_size, 0);
    
    printf("[kernel config]\n");
    printf("tile: %d | regs/thread: %d | shared/block: %zu bytes\n", 
           tile, attr.numRegs, attr.sharedSizeBytes);
    printf("max blocks per sm: %d | max threads per sm: %d\n\n", 
           max_blocks, max_blocks * block_size);
}

void run_test(int dim, int tile) {
    printf("\n>> testing dim: %d, tile: %d\n", dim, tile);
    
    float *host_a, *host_b, *host_res;
    host_a = (float*)malloc(dim*dim*sizeof(float));
    host_b = (float*)malloc(dim*dim*sizeof(float));
    host_res = (float*)malloc(dim*dim*sizeof(float));
    
    init_rand_matrix(host_a, dim);
    init_rand_matrix(host_b, dim);
    
    float *dev_a, *dev_b, *dev_res;
    cudaMalloc(&dev_a, dim*dim*sizeof(float));
    cudaMalloc(&dev_b, dim*dim*sizeof(float));
    cudaMalloc(&dev_res, dim*dim*sizeof(float));
    
    cudaMemcpy(dev_a, host_a, dim*dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, dim*dim*sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 block(tile, tile);
    dim3 grid((dim + tile -1)/tile, (dim + tile -1)/tile);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    gpu_mmul_tiled<<<grid, block>>>(dev_a, dev_b, dev_res, dim, tile);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    
    cudaMemcpy(host_res, dev_res, dim*dim*sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("gpu time: %.2f ms | ", gpu_ms);
    printf("verification: %s\n", verify_result(host_a, host_b, host_res, dim) ? "test PASSED" : "Test FAILED");
    
    free(host_a); free(host_b); free(host_res);
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_res);
}

int main() {
    srand(time(NULL));
    
    int tiles[] = {2,4,8,16,32};
    int dims[] = {256,512,1024,2048,4096};
    
    for (int t = 0; t < 5; t++) {
        get_kernel_stats(tiles[t]);
    }
    
    for (int t = 0; t < 5; t++) {
        for (int d = 0; d < 5; d++) {
            run_test(dims[d], tiles[t]);
        }
    }
    
    return 0;
}
