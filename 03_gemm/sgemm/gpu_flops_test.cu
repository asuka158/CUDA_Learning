#include <stdio.h>
#include <cuda_runtime.h>

#define N (1 << 27) // 128M数据，适中
#define THREADS_PER_BLOCK 256

__global__ void flops_kernel(float *a, float *b, float *c, int n, int repeat) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        float y = b[i];
        float z = 0.0f;
        // repeat保证每个线程做更多浮点运算
        for (int j = 0; j < repeat; j++) {
            // 8个浮点运算/循环
            x = x * 1.000001f + y;
            y = y * 0.999999f + x;
            z = z + x * y;
        }
        c[i] = z;
    }
}

int main() {
    int repeat = 4096; // 每个线程重复次数（可增大以拉长时间提高精度）
    int n = N;

    float *a, *b, *c;
    float *da, *db, *dc;

    size_t size = n * sizeof(float);
    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);

    // 初始化
    for (int i = 0; i < n; ++i) {
        a[i] = 1.1f;
        b[i] = 2.2f;
    }

    cudaMalloc(&da, size);
    cudaMalloc(&db, size);
    cudaMalloc(&dc, size);

    cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, size, cudaMemcpyHostToDevice);

    dim3 threads(THREADS_PER_BLOCK);
    dim3 blocks((n + threads.x - 1) / threads.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    flops_kernel<<<blocks, threads>>>(da, db, dc, n, repeat);
    cudaDeviceSynchronize();

    cudaEventRecord(start, 0);
    flops_kernel<<<blocks, threads>>>(da, db, dc, n, repeat);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // 每次循环有8次FLOP
    double total_flops = (double)n * repeat * 6;
    double gflops = total_flops / (ms * 1e6);

    printf("FP32 FLOPS: %.2f GFLOPS (%.2f ms, %d threads, repeat %d)\n", gflops, ms, n, repeat);

    cudaMemcpy(c, dc, size, cudaMemcpyDeviceToHost);

    cudaFree(da); cudaFree(db); cudaFree(dc);
    free(a); free(b); free(c);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}