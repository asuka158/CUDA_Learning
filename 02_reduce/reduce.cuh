#pragma once
#include <cuda_runtime.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const unsigned FULL_MASK = 0xffffffff;

void __global__ reduce_global(real *d_x, real *d_y)
{
    //还是放在寄存器，不是常量内存，const跟__constant__是不一样的
    //const修饰是可以让编译器做一些优化的
    const int tid = threadIdx.x; 
    const int bid = blockIdx.x;
    real *x = d_x + blockIdx.x * blockDim.x;

    for(int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if(tid < offset) x[tid] += x[tid + offset];
        __syncthreads();
    }

    if(tid == 0) d_y[bid] = x[0];
}

void __global__ reduce_shared(real *d_x, real *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ real s_y[128];
    s_y[tid] = (idx < N) ? d_x[idx] : 0.0;
    __syncthreads();

    for(int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if(tid < offset) s_y[tid] += s_y[tid + offset];
        __syncthreads();
    }

    if(tid == 0) d_y[bid] = s_y[0];
}

void __global__ reduce_dynamic(real *d_x, real *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ real s_y[];
    s_y[tid] = (idx < N) ? d_x[idx] : 0.0;
    __syncthreads();

    for(int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if(tid < offset) s_y[tid] += s_y[tid + offset];
        __syncthreads();
    }

    if(tid == 0) d_y[bid] = s_y[0];
}

void __global__ reduce_atomic(real *d_x, real *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ real s_y[];
    s_y[tid] = (idx < N) ? d_x[idx] : 0.0;
    __syncthreads();

    for(int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if(tid < offset) s_y[tid] += s_y[tid + offset];
        __syncthreads();
    }

    if(tid == 0) atomicAdd(d_y, s_y[0]);
}

void __global__ reduce_syncwarp(real *d_x, real *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ real s_y[];
    s_y[tid] = (idx < N) ? d_x[idx] : 0.0;
    __syncthreads();

    for(int offset = blockDim.x >> 1; offset > 32; offset >>= 1)
    {
        if(tid < offset) s_y[tid] += s_y[tid + offset];
        __syncthreads();
    }

    // 依旧存在线程束分化，循环展开 or 洗牌函数才能彻底解决
    if(tid < 32) 
    {
        for(int offset = 32; offset > 0; offset >>= 1)
        {
            if(tid < offset) s_y[tid] += s_y[tid + offset];
            __syncwarp();
        }
    }

    if(tid == 0) atomicAdd(d_y, s_y[0]);
}

void __global__ reduce_shfl(real *d_x, real *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ real s_y[];
    s_y[tid] = (idx < N) ? d_x[idx] : 0.0;
    __syncthreads();

    for(int offset = blockDim.x >> 1; offset > 16; offset >>= 1)
    {
        if(tid < offset) s_y[tid] += s_y[tid + offset];
        __syncthreads();
    }

    real y = s_y[tid];
    if(tid < 32) 
    {
        for(int offset = 16; offset > 0; offset >>= 1)
            y += __shfl_down_sync(FULL_MASK, y, offset);
    }

    if(tid == 0) atomicAdd(d_y, y);
}

void __global__ reduce_cp(real *d_x, real *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ real s_y[];
    s_y[tid] = (idx < N) ? d_x[idx] : 0.0;
    __syncthreads();

    for(int offset = blockDim.x >> 1; offset > 16; offset >>= 1)
    {
        if(tid < offset) s_y[tid] += s_y[tid + offset];
        __syncthreads();
    }

    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    real y = s_y[tid];
    if(tid < 32) 
    {
        for(int offset = g.size() >> 1; offset > 0; offset >>= 1)
            y += g.shfl_down(y, offset);
    }

    if(tid == 0) atomicAdd(d_y, y);
}

void __global__ reduce_idle(real *d_x, real *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ real s_y[];

    real y = 0.0;
    const int stride = blockDim.x * gridDim.x;
    for(int i = idx; i < N; i += stride) y += d_x[i];
    s_y[tid] = y;
    __syncthreads();

    for(int offset = blockDim.x >> 1; offset > 16; offset >>= 1)
    {
        if(tid < offset) s_y[tid] += s_y[tid + offset];
        __syncthreads();
    }

    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    y = s_y[tid];
    
    if(tid < 32) 
    {
        for(int offset = g.size() >> 1; offset > 0; offset >>= 1)
            y += g.shfl_down(y, offset);
    }

    if(tid == 0) atomicAdd(d_y, y);
}