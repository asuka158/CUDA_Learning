#include <float.h>

#define CEIL(a, b) ((a + b - 1) / (b))
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

void softmax(float* input, float* output, int N)
{
    float M = *(std::max_element(input, input + N));
    float div = 0;
    
    for(int i = 0; i < N; ++i)
    {
        output[i] = std::exp(input[i] - M);
        div += output[i];
    }

    for(int i = 0; i < N; ++i) output[i] /= div;
}

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*)address;
    int old = *address_as_i;
    int assumed;
    do{
        assumed = old;
        old = atomicCAS(
            address_as_i, 
            assumed, 
            __float_as_int(fmaxf(val, __int_as_float(assumed)))
        );
    } while(assumed != old);
    return __int_as_float(old);
}

__global__ void max_kernel(float* input, float* max_val, int N)
{
    __shared__ float s_mem[32];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x & (warpSize - 1);

    float val = (idx < N) ? input[idx] : (-FLT_MAX);
    for(int offset = warpSize >> 1; offset > 0; offset >>= 1) 
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));

    if(laneId == 0) s_mem[warpId] = val;
    __syncthreads();

    if(warpId == 0)
    {
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum) ? s_mem[laneId] : (-FLT_MAX);
        for(int offset = warpSize >> 1; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        if(laneId == 0) atomicMax(max_val, val);    
    }
}

__global__ void sum_kernel(float* input, float* sum, float* max_val, int N)
{
    __shared__ float s_mem[32];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x & (warpSize - 1);

    float val = (idx < N) ? expf(input[idx] - *max_val) : 0.0f;
    for(int offset = warpSize >> 1; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    if(laneId == 0) s_mem[warpId] = val;
    __syncthreads();

    if(warpId == 0)
    {
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum) ? s_mem[laneId] : 0.0f;
        for(int offset = warpSize >> 1; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if(laneId == 0) atomicAdd(sum, val);
    }
}

__global__ void softmax_kernel(float* input, float* output, float* sum, float* max_val, const int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N) output[idx] = expf(input[idx] - *max_val) / (*sum);
}

int block_size = 256;
int grid_size = CEIL(N, ,block_size);
max_kernel<<<grid_size, block_size>>>(input, max_val, N);
sum_kernel<<<grid_size, block_size>>>(input, sum, max_val, N);
softmax_kernel<<<grid_size, block_size>>>(input, output, sum, max_val, N);
