#include "hgemm.cuh"
#include <stdio.h>
#include <stdlib.h>

using namespace hgemm;

typedef enum{
    HGEMMAlignedV1,
    HGEMMAlignedV2,
    HGEMMAlignedV3,
    HGEMMAlignedV4,
    HGEMMAlignedV5
} F16F16GemmTCAlgo_t;

void cpuF16F16Gemm(half *a, half *b, half *c, int M, int N, int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += (float)a[OFFSET(m, k, K)] * (float)b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = (half)psum;
        }
    }
}

template<F16F16GemmTCAlgo_t algo = HGEMMAlignedV1>
void myF16F16GemmTCWarp(half *a, half *b, half *c, int M, int N, int K) {

    if (algo == HGEMMAlignedV1) {
        const int BM = 128, BN = 256;
        dim3 blockDim(256);
        int BX = (N + BN - 1) / BN;
        int BY = (M + BM - 1) / BM;
        dim3 gridDim(BX, BY);
        myHGEMMAlignedV1<<<gridDim, blockDim>>>(a, b, c, M, N, K);
    }
    else if (algo == HGEMMAlignedV2) {
        const int BM = 128, BN = 256;
        dim3 blockDim(256);
        int BX = (N + BN - 1) / BN;
        int BY = (M + BM - 1) / BM;
        dim3 gridDim(BX, BY);
        myHGEMMAlignedV2<<<gridDim, blockDim>>>(a, b, c, M, N, K);
    }
    else if (algo == HGEMMAlignedV3) {
        const int BM = 128, BN = 256, BK = 32;
        dim3 blockDim(256);
        int BX = (N + BN - 1) / BN;
        int BY = (M + BM - 1) / BM;
        dim3 gridDim(BX, BY);

        cudaFuncSetAttribute(myHGEMMAlignedV3,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        unsigned int dsmem = 2 * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(half);
        myHGEMMAlignedV3<<<gridDim, blockDim, dsmem>>>(a, b, c, M, N, K);
    }
    else if (algo == HGEMMAlignedV4) {
        const int BM = 128, BN = 256, BK = 32;
        dim3 blockDim(256);
        int BX = (N + BN - 1) / BN;
        int BY = (M + BM - 1) / BM;

        const int NSPLIT = 4096;
        int split_num = (N + NSPLIT - 1) / NSPLIT;
        dim3 gridDim((BX + split_num - 1) / split_num, BY, split_num);

        cudaFuncSetAttribute(myHGEMMAlignedV4,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        unsigned int dsmem = 2 * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(half);
        myHGEMMAlignedV4<<<gridDim, blockDim, dsmem>>>(a, b, c, M, N, K);
    }
    else if (algo == HGEMMAlignedV5) {
        const int BM = 128, BN = 256, BK = 32;
        dim3 blockDim(256);
        int BX = (N + BN - 1) / BN;
        int BY = (M + BM - 1) / BM;

        const int NSPLIT = 4096;
        int split_num = (N + NSPLIT - 1) / NSPLIT;
        dim3 gridDim((BX + split_num - 1) / split_num, BY, split_num);

        cudaFuncSetAttribute(myHGEMMAlignedV5,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        unsigned int dsmem = 2 * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(half);
        myHGEMMAlignedV5<<<gridDim, blockDim, dsmem>>>(a, b, c, M, N, K);
    }
}

float testF16F16GemmMaxError(
    void (*gpuF16F16Gemm) (half *, half *, half *, int, int, int),
    int M, int N, int K) {

    size_t size_a = M * K * sizeof(half);
    size_t size_b = K * N * sizeof(half);
    size_t size_c = M * N * sizeof(half);

    half *h_a, *h_b, *d_a, *d_b;
    half *h_c, *d_c, *h_d_c;
    h_a = (half *)malloc(size_a);
    h_b = (half *)malloc(size_b);
    h_c = (half *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (half *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = (half)(rand() / float(RAND_MAX));
    for (int i = 0; i < K * N; i++)
        h_b[i] = (half)(rand() / float(RAND_MAX));

    cpuF16F16Gemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuF16F16Gemm(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs((float)h_d_c[i] - (float)h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); free(h_d_c);

    return max_error;
}

float testF16F16GemmPerformance(
    void (*gpuF16F16Gemm) (half *, half *, half *, int, int, int),
    int M, int N, int K, int repeat) {

    size_t size_a = M * K * sizeof(half);
    size_t size_b = K * N * sizeof(half);
    size_t size_c = M * N * sizeof(half);

    half *d_a, *d_b;
    half *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        gpuF16F16Gemm(d_a, d_b, d_c, M, N, K);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return sec;
}

int main() {

    /*
    const int test_num = 7;
    const int M_list[test_num] = {256, 512, 1024, 2048, 4096, 8192, 16384};
    const int N_list[test_num] = {256, 512, 1024, 2048, 4096, 8192, 16384};
    const int K_list[test_num] = {256, 512, 1024, 2048, 4096, 8192, 16384};
    */

    const int test_num = 64;
    int M_list[test_num];
    int N_list[test_num];
    int K_list[test_num];
    for (int i = 0; i < test_num; i++) {
        M_list[i] = (i + 1) * 256;
        N_list[i] = (i + 1) * 256;
        K_list[i] = (i + 1) * 256;
    }

    const int outer_repeat = 10, inner_repeat = 1;

    {
        printf("\nalgo = HGEMMAlignedV1\n");

        {
            const int M = 256, N = 256, K = 256;
            float max_error = testF16F16GemmMaxError(
                myF16F16GemmTCWarp<HGEMMAlignedV1>, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        for (int j = 0; j < test_num; j++) {
            int M = M_list[j], N = N_list[j], K = K_list[j];

            double max_sec = 0.0;
            double min_sec = DBL_MAX;
            double total_sec = 0.0;

            for (int k = 0; k < outer_repeat; k++) {
                double this_sec = testF16F16GemmPerformance(
                    myF16F16GemmTCWarp<HGEMMAlignedV1>, M, N, K, inner_repeat);
                max_sec = max(max_sec, this_sec);
                min_sec = min(min_sec, this_sec);
                total_sec += this_sec;
            }

            double avg_sec = total_sec / outer_repeat;
            double avg_Gflops = ((double)M) * N * K * 2 / 1000 / 1000 / 1000 / avg_sec;

            printf("M N K = %6d %6d %6d, ", M, N, K);
            printf("Time = %12.8lf %12.8lf %12.8lf s, ", min_sec, avg_sec, max_sec);
            printf("AVG Performance = %10.4lf Gflops\n", avg_Gflops);
        }
    }

    {
        printf("\nalgo = HGEMMAlignedV2\n");

        {
            const int M = 256, N = 256, K = 256;
            float max_error = testF16F16GemmMaxError(
                myF16F16GemmTCWarp<HGEMMAlignedV2>, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        for (int j = 0; j < test_num; j++) {
            int M = M_list[j], N = N_list[j], K = K_list[j];

            double max_sec = 0.0;
            double min_sec = DBL_MAX;
            double total_sec = 0.0;

            for (int k = 0; k < outer_repeat; k++) {
                double this_sec = testF16F16GemmPerformance(
                    myF16F16GemmTCWarp<HGEMMAlignedV2>, M, N, K, inner_repeat);
                max_sec = max(max_sec, this_sec);
                min_sec = min(min_sec, this_sec);
                total_sec += this_sec;
            }

            double avg_sec = total_sec / outer_repeat;
            double avg_Gflops = ((double)M) * N * K * 2 / 1000 / 1000 / 100 / avg_sec;

            printf("M N K = %6d %6d %6d, ", M, N, K);
            printf("Time = %12.8lf %12.8lf %12.8lf s, ", min_sec, avg_sec, max_sec);
            printf("AVG Performance = %10.4lf Gflops\n", avg_Gflops);
        }
    }

    {
        printf("\nalgo = HGEMMAlignedV3\n");

        {
            const int M = 256, N = 256, K = 256;
            float max_error = testF16F16GemmMaxError(
                myF16F16GemmTCWarp<HGEMMAlignedV3>, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        for (int j = 0; j < test_num; j++) {
            int M = M_list[j], N = N_list[j], K = K_list[j];

            double max_sec = 0.0;
            double min_sec = DBL_MAX;
            double total_sec = 0.0;

            for (int k = 0; k < outer_repeat; k++) {
                double this_sec = testF16F16GemmPerformance(
                    myF16F16GemmTCWarp<HGEMMAlignedV3>, M, N, K, inner_repeat);
                max_sec = max(max_sec, this_sec);
                min_sec = min(min_sec, this_sec);
                total_sec += this_sec;
            }

            double avg_sec = total_sec / outer_repeat;
            double avg_Gflops = ((double)M) * N * K * 2 / 1000 / 1000 / 1000 / avg_sec;

            printf("M N K = %6d %6d %6d, ", M, N, K);
            printf("Time = %12.8lf %12.8lf %12.8lf s, ", min_sec, avg_sec, max_sec);
            printf("AVG Performance = %10.4lf Gflops\n", avg_Gflops);
        }
    }

    {
        printf("\nalgo = HGEMMAlignedV4\n");

        {
            const int M = 256, N = 256, K = 256;
            float max_error = testF16F16GemmMaxError(
                myF16F16GemmTCWarp<HGEMMAlignedV4>, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        for (int j = 0; j < test_num; j++) {
            int M = M_list[j], N = N_list[j], K = K_list[j];

            double max_sec = 0.0;
            double min_sec = DBL_MAX;
            double total_sec = 0.0;

            for (int k = 0; k < outer_repeat; k++) {
                double this_sec = testF16F16GemmPerformance(
                    myF16F16GemmTCWarp<HGEMMAlignedV4>, M, N, K, inner_repeat);
                max_sec = max(max_sec, this_sec);
                min_sec = min(min_sec, this_sec);
                total_sec += this_sec;
            }

            double avg_sec = total_sec / outer_repeat;
            double avg_Gflops = ((double)M) * N * K * 2 / 1000 / 1000 / 1000 / avg_sec;

            printf("M N K = %6d %6d %6d, ", M, N, K);
            printf("Time = %12.8lf %12.8lf %12.8lf s, ", min_sec, avg_sec, max_sec);
            printf("AVG Performance = %10.4lf Gflops\n", avg_Gflops);
        }
    }

    {
        printf("\nalgo = HGEMMAlignedV5\n");

        {
            const int M = 256, N = 256, K = 256;
            float max_error = testF16F16GemmMaxError(
                myF16F16GemmTCWarp<HGEMMAlignedV5>, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        for (int j = 0; j < test_num; j++) {
            int M = M_list[j], N = N_list[j], K = K_list[j];

            double max_sec = 0.0;
            double min_sec = DBL_MAX;
            double total_sec = 0.0;

            for (int k = 0; k < outer_repeat; k++) {
                double this_sec = testF16F16GemmPerformance(
                    myF16F16GemmTCWarp<HGEMMAlignedV5>, M, N, K, inner_repeat);
                max_sec = max(max_sec, this_sec);
                min_sec = min(min_sec, this_sec);
                total_sec += this_sec;
            }

            double avg_sec = total_sec / outer_repeat;
            double avg_Gflops = ((double)M) * N * K * 2 / 1000 / 1000 / 1000 / avg_sec;

            printf("M N K = %6d %6d %6d, ", M, N, K);
            printf("Time = %12.8lf %12.8lf %12.8lf s, ", min_sec, avg_sec, max_sec);
            printf("AVG Performance = %10.4lf Gflops\n", avg_Gflops);
        }
    }
}
