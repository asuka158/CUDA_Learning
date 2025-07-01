#pragma once
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

namespace sgemm{
    
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    void cpuSgemm(float *a, float *b, float *c, const int M, const int N, const int K)
    {
        for (int m = 0; m < M; ++m) 
        {
            for (int n = 0; n < N; ++n) 
            {
                float psum = 0.0;
                for (int k = 0; k < K; ++k) psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
                c[OFFSET(m, n, N)] = psum;
            }
        }
    }
    
    //__restrict__ 的作用​​:开发者向编译器保证：在该指针的作用域内，所有通过该指针访问的内存不会通过其他指针或引用被修改。编译器可以据此假设无别名冲突，从而生成更高效的代码。
    __global__ void naiveSgemm(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, const int M, const int N, const int K)
    {
        int n = blockIdx.x * blockDim.x + threadIdx.x;
        int m = blockIdx.y * blockDim.y + threadIdx.y;
        
        if(n < N && m < M)
        {
            float sum = 0.0;
            #pragma unroll 
            for(int k = 0; k < K; ++k) 
            {
                sum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = sum;
        }
    }


    __global__ void sgemm_V1(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c, const int M, const int N, const int K)
    {
        const int bx = blockIdx.x;
        const int by = blockIdx.y;
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int tid = ty * blockDim.x + tx;

        __shared__ float s_a[BM][BK];
        __shared__ float s_b[BK][BN];
        
        float r_c[TM][TN] = {0.0};

        // 加载 到 s_a 的 m 行
        int load_a_smem_m = tid >> 1;
        int load_a_smem_k = (tid & 1) << 2;
        int load_b_smem_k = tid >> 5;
        int load_b_smem_n = (tid & 31) << 2;

        // 从 g_a 的 m 行 加载
        int load_a_gmem_m = by * BM + load_a_smem_m;
        int load_b_gmem_n = bx * BN + load_b_smem_n;

        int cnt = (K + BK - 1) / BK;
        for(int bk = 0; bk < cnt; ++bk)
        {
            int load_a_gmem_k = bk * BK + load_a_smem_k;
            int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
            FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);
            int load_b_gmem_k = bk * BK + load_b_smem_k;
            int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
            FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr]);

            __syncthreads();

            #pragma unroll
            for(int k = 0; k < BK; ++k)
            {
                #pragma unroll
                for(int m = 0; m < TM; ++m)
                {
                    #pragma unroll
                    for(int n = 0; n < TN; ++n)
                    {
                        int comp_a_smem_m = ty * TM + m;
                        int comp_b_smem_n = tx * TN + n;
                        r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
                    }
                }
            }

            __syncthreads();
        }

        #pragma unroll
        for(int i = 0; i < TM; ++i)
        {
            int store_c_gmem_m = by * BM + ty * TM + i;
            #pragma unroll
            for(int j = 0; j < TN; j += 4)
            {
                int store_c_gmem_n = bx * BN + tx * TN + j;
                int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
                FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);
            }
        }
    }

    // 尝试将 寄存器 全部改为 float4
    __global__ void sgemm_V2(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c, const int M, const int N, const int K)
    {
        const int bx = blockIdx.x;
        const int by = blockIdx.y;
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int tid = ty * blockDim.x + tx;

        __shared__ float s_a[BK][BM];
        __shared__ float s_b[BK][BN];

        float r_c[TN][TN] = {0.0};
        float r_load_a[4];
        float r_load_b[4];
        float r_comp_a[TM];
        float r_comp_b[TN];

        int load_a_smem_m = tid >> 1;
        int load_a_smem_k = (tid & 1) << 2;
        int load_b_smem_k = tid >> 5;
        int load_b_smem_n = (tid & 31) << 2;

        int load_a_gmem_m = by * BM + load_a_smem_m;
        int load_b_gmem_n = bx * BN + load_b_smem_n;

        int cnt = (K + BK - 1) / BK;
        for(int bk = 0; bk < cnt; ++bk)
        {
            int load_a_gmem_k = bk * BK + load_a_smem_k;
            int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
            int load_b_gmem_k = bk * BK + load_b_smem_k;
            int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
            FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
            FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

            s_a[load_a_smem_k][load_a_smem_m] = r_load_a[0];
            s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
            s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
            s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
            FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

            __syncthreads();

            #pragma unroll
            for(int k = 0; k < BK; ++k)
            {
                FLOAT4(r_comp_a[0]) = FLOAT4(s_a[k][ty * TM / 2]);
                FLOAT4(r_comp_a[4]) = FLOAT4(s_a[k][ty * TM / 2 + BM / 2]);
                FLOAT4(r_comp_b[0]) = FLOAT4(s_b[k][tx * TN / 2]);
                FLOAT4(r_comp_b[4]) = FLOAT4(s_b[k][tx * TN / 2 + BN / 2]);
                
                #pragma unroll
                for(int m = 0; m < TM; ++m)
                {
                    #pragma unroll
                    for(int n = 0; n < TN; ++n) r_c[m][n] += r_comp_a[m] * r_comp_b[n];
                }
            }

            __syncthreads();
        }

        #pragma unroll
        for(int i = 0; i < TM / 2; ++i)
        {
            int store_c_gmem_m = by * BM + ty * TM / 2 + i;
            int store_c_gmem_n = bx * BN + tx * TN / 2;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
            FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
        }
        #pragma unroll
        for(int i = 0; i < TM / 2; ++i)
        {
            int store_c_gmem_m = by * BM + ty * TM / 2 + i + BM / 2;
            int store_c_gmem_n = bx * BN + tx * TN / 2;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
            FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
        }
        
    }

    __global__ void sgemm_V3(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c, const int M, const int N, const int K)
    {
        const int bx = blockIdx.x;
        const int by = blockIdx.y;
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int tid = ty * blockDim.x + tx;

        __shared__ float s_a[2][BK][BM];
        __shared__ float s_b[2][BK][BN];

        float r_c[TM][TN] = {0.0};
        float r_load_a[4];
        float r_load_b[4];
        float r_comp_a[TM];
        float r_comp_b[TN];

        int load_a_smem_m = tid >> 1;
        int load_a_smem_k = (tid & 1) << 2;
        int load_b_smem_k = tid >> 5;
        int load_b_smem_n = (tid & 31) << 2;

        int load_a_gmem_m = by * BM + load_a_smem_m;
        int load_b_gmem_n = bx * BN + load_b_smem_n;

        {
            int load_a_gmem_k = load_a_smem_k;
            int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
            int load_b_gmem_k = load_b_smem_k;
            int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
            FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
            FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

            s_a[0][load_a_smem_k][load_a_smem_m] = r_load_a[0];
            s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
            s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
            s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
            FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
        }

        
        int cnt = (K + BK - 1) / BK;
        for(int bk = 1; bk < cnt; ++bk)
        {
            int smem_sel = (bk - 1) & 1;
            int smem_sel_next = bk & 1;

            int load_a_gmem_k = bk * BK + load_a_smem_k;
            int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
            int load_b_gmem_k = bk * BK + load_b_smem_k;
            int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
            FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
            FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

            #pragma unroll
            for(int k = 0; k < BK; ++k)
            {
                FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][k][ty * TM / 2]);
                FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][k][ty * TM / 2 + BM / 2]);
                FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][k][tx * TN / 2]);
                FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][k][tx * TN / 2 + BN / 2]);

                #pragma unroll
                for(int m = 0; m < TM; ++m)
                {
                    #pragma unroll
                    for(int n = 0; n < TN; ++n) r_c[m][n] += r_comp_a[m] * r_comp_b[n];
                }
            }

            s_a[smem_sel_next][load_a_smem_k][load_a_smem_m] = r_load_a[0];
            s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
            s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
            s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
            FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

            __syncthreads();
        }

        #pragma unroll
        for(int k = 0; k < BK; ++k)
        {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][k][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][k][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][k][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[1][k][tx * TN / 2 + BN / 2]);

            #pragma unroll
            for(int m = 0; m < TM; ++m)
            {
                #pragma unroll
                for(int n = 0; n < TN; ++n) r_c[m][n] += r_comp_a[m] * r_comp_b[n];
            }
        }
        
        #pragma unroll
        for(int i = 0; i < TM / 2; ++i)
        {
            int store_c_gmem_m = by * BM + ty * TM / 2 + i;
            int store_c_gmem_n = bx * BN + tx * TN / 2;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
            FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
        }

        #pragma unroll
        for(int i = 0; i < TM / 2; ++i)
        {
            int store_c_gmem_m = by * BM + ty * TM / 2 + i + BM / 2;
            int store_c_gmem_n = bx * BN + tx * TN / 2;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
            FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
        }
        
    }

}

