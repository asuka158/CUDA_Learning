#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <ctime>
#include <iostream>
#include <vector>
#include <random>
#include <cuda_bf16.h>
#include <cassert>
#include <unistd.h>
#include <fstream>

#include "examples/matmul/matmul_1.cuh"
#include "examples/matmul/matmul_2.cuh"
// #include "examples/matmul/matmul_3.cuh"
// #include "examples/matmul/matmul_4.cuh"
// #include "examples/matmul/matmul_5.cuh"
// #include "examples/matmul/matmul_6.cuh"
// #include "examples/matmul/matmul_7.cuh"
// #include "examples/matmul/matmul_8.cuh"
// #include "examples/matmul/matmul_9.cuh"
// #include "examples/matmul/matmul_10.cuh"
// #include "examples/matmul/matmul_11.cuh"
// #include "examples/matmul/matmul_12.cuh"

#define OFFSET(r, c, ld) ((r) * (ld) + (c))

#define checkCudaErrors(func) do {                                      \
    cudaError_t e = (func);                                             \
    if (e != cudaSuccess) {                                             \
        printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1);                                                        \
    }                                                                   \
} while (0)

#define checkCublasErrors(func) do {                                    \
    cublasStatus_t e = (func);                                          \
    if (e != CUBLAS_STATUS_SUCCESS) {                                   \
        printf("%s %d cuBLAS: %d\n", __FILE__, __LINE__, (int)e);       \
        exit(1);                                                        \
    }                                                                   \
} while (0)

typedef __nv_bfloat16 bf16;

// cpu
void cpuBF16Gemm(const bf16* a, const bf16* b, bf16* c, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {

            float acc = 0.0f;

            for (int k = 0; k < K; ++k) {
                float av = __bfloat162float(a[OFFSET(m,k,K)]);
                float bv = __bfloat162float(b[OFFSET(k,n,N)]);
                acc += av * bv;
            }

            c[OFFSET(m,n,N)] = __float2bfloat16(acc);
        }
    }
}

// 手写
void run_kernel(int kernel_num, int M, int N, int K, bf16 *A, bf16 *B, bf16 *C, int *DB = nullptr)
{
    switch (kernel_num) {
      case 0:
        // runCublasGemmBF16(M, N, K, A, B, C);
        break;
      case 1:
        runKernel1(M, N, K, A, B, C);
        break;
      case 2:
        runKernel2(M, N, K, A, B, C);
        break;
      // case 3:
      //   runKernel3(M, N, K, A, B, C, DB);
      //   break;
      // case 4:
      //   runKernel4(M, N, K, A, B, C, DB);
      //   break;
      // case 5:
      //   runKernel5(M, N, K, A, B, C, DB);
      //   break;
      // case 6:
      //   runKernel6(M, N, K, A, B, C, DB);
      //   break;
      // case 7:
      //   runKernel7(M, N, K, A, B, C, DB);
      //   break;
      // case 8:
      //   runKernel8(M, N, K, A, B, C, DB);
      //   break;
      // case 9:
      //   runKernel9(M, N, K, A, B, C, DB);
      //   break;
      // case 10:
      //   runKernel10(M, N, K, A, B, C, DB);
      //   break;
      // case 11:
      //   runKernel11(M, N, K, A, B, C, DB);
      //   break;
      // case 12:
      //   runKernel12(M, N, K, A, B, C, DB);
      //   break;
    }
}

// cublas
template <cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP>
void cublasBF16Gemm(const bf16* a, const bf16* b, bf16* c,
                    int M, int N, int K)
{
    cublasHandle_t handle;
    checkCublasErrors(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta  = 0.0f;

    checkCublasErrors(
        cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,

            /*m=*/N,
            /*n=*/M,
            /*k=*/K,

            &alpha,

            /*A=*/b,
            CUDA_R_16BF,
            /*lda=*/N,

            /*B=*/a,
            CUDA_R_16BF,
            /*ldb=*/K,

            &beta,

            /*C=*/c,
            CUDA_R_16BF,
            /*ldc=*/N,

            CUBLAS_COMPUTE_32F,
            algo
        )
    );

    cublasDestroy(handle);
}

void saveBF16MatrixCSV(const char* filename,
    const bf16* data,
    int rows,
    int cols,
    int ld)
{
    std::ofstream file(filename);

    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            float v = __bfloat162float(data[i * ld + j]);
            file << v;

            if(j != cols - 1)
                file << ",";
        }
        file << "\n";
    }

    file.close();
}

// 测精度
float testBF16GemmMaxError(int fun, int M, int N, int K)
{
    size_t size_a = (size_t)M*K*sizeof(bf16);
    size_t size_b = (size_t)K*N*sizeof(bf16);
    size_t size_c = (size_t)M*N*sizeof(bf16);

    bf16 *h_a = (bf16*)malloc(size_a);
    bf16 *h_b = (bf16*)malloc(size_b);
    bf16 *h_c = (bf16*)malloc(size_c);
    bf16 *h_d_c = (bf16*)malloc(size_c);

    bf16 *d_a,*d_b,*d_c;

    checkCudaErrors(cudaMalloc(&d_a,size_a));
    checkCudaErrors(cudaMalloc(&d_b,size_b));
    checkCudaErrors(cudaMalloc(&d_c,size_c));

    srand(time(0));

    for(int i = 0; i < M; ++i)
    {
        int of = rand();
        for(int j = 0; j < K; ++j)
        {
            h_a[i * K + j] = __float2bfloat16((of + i + j) % 16);
        }
    }

    for(int i = 0; i < K; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            h_b[i * N + j] = __float2bfloat16(15 - ((i + j) % 16));
        }
    }

    cpuBF16Gemm(h_a,h_b,h_c,M,N,K);

    /* ---------------- 保存 CSV ---------------- */

    // A 的前128行
    saveBF16MatrixCSV("./test/a.csv", h_a, min(128, M), K, K);

    // B 的前128列
    saveBF16MatrixCSV("./test/b.csv", h_b, K, min(128, N), N);

    // CPU C 左上角128x128
    saveBF16MatrixCSV("./test/cpu_c.csv", h_c, min(128, M), min(128, N), N);

    /* ------------------------------------------ */

    cudaMemcpy(d_a,h_a,size_a,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size_b,cudaMemcpyHostToDevice);

    run_kernel(1, M, N, K, d_a, d_b, d_c);

    cudaMemcpy(h_d_c,d_c,size_c,cudaMemcpyDeviceToHost);

    // GPU C 左上角128x128
    saveBF16MatrixCSV("./test/gpu_c.csv", h_d_c, min(128, M), min(128, N), N);

    float max_err=0;

    for(int i = 0; i < M; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            float ref = __bfloat162float(h_c[i * N + j]);
            float val = __bfloat162float(h_d_c[i * N + j]);

            float diff = fabs(ref - val);

            if(diff > max_err)
                max_err = diff;
        }
    }

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return max_err;
}

// 包装
// void cublasI8I8Gemm_default(const int8_t* a, const int8_t* b, int32_t* c, int M, int N, int K) {
//     cublasI8I8Gemm<>(a, b, c, M, N, K);
// }

int main() {
    const int M = 128, N = 128, K = 128;
    float max_error = testBF16GemmMaxError(2, M, N, K);
    printf("Max Error = %.0f\n", max_error);
    return 0;
}