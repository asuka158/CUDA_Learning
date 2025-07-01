#include <stdio.h>
#include <cuda_runtime.h>

// 线程块大小
#define BLOCK_SIZE 256
// 总数据量（单位：字节）
#define TOTAL_SIZE_MB 1024
#define TOTAL_SIZE (TOTAL_SIZE_MB * 1024 * 1024)

__global__ void global_mem_read(float *src, float *dst, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t idx = i; idx < n; idx += stride) {
        dst[idx] = src[idx];
    }
}

int main() {
    printf("CUDA Global Memory Bandwidth Test (%d MB)\n", TOTAL_SIZE_MB);

    float *d_src, *d_dst;
    size_t num_floats = TOTAL_SIZE / sizeof(float);

    // 设备分配
    cudaMalloc(&d_src, TOTAL_SIZE);
    cudaMalloc(&d_dst, TOTAL_SIZE);

    // 填充src
    cudaMemset(d_src, 1, TOTAL_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = 256; // 可根据GPU调整
    int threads = BLOCK_SIZE;

    // warm up
    global_mem_read<<<blocks, threads>>>(d_src, d_dst, num_floats);
    cudaDeviceSynchronize();

    cudaEventRecord(start, 0);
    global_mem_read<<<blocks, threads>>>(d_src, d_dst, num_floats);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // 总访问字节数 = 读 + 写
    double bytes = 2.0 * TOTAL_SIZE; // 读和写各一次
    double bandwidth = bytes / (ms * 1e6); // GB/s

    printf("Global Memory Bandwidth: %.2f GB/s (%.2f ms)\n", bandwidth, ms);

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}