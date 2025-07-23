#define CEIL(a, b) ((a + b - 1) / (b))
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

dim3 block_size(BLOCK_SIZE);
dim3 grid_size(CIEL(N, BLOCK_SIZE));

__global__ void reduce_v1(const float* input, float* output, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N) atomicAdd(output, input[idx]);
}

__global__ void reduce_v2(const float* input, float* output, int N)
{
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    __shared__ float input_s[BLOCK_SIZE];

    input_s[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    for(int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if(tid < offset) input_s[tid] += input_s[tid + offset];
        __syncthreads();
    }

    if(tid == 0) atomicAdd(output, input_s[0]);
}

__global__ void reduce_v3(float* d_x, float* d_y, const int N)
{
    __shared__ float s_y[32];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x & (warpSize - 1);

    float val = (idx < N) ? d_x[idx] : 0.0f;
    
    #pragma unroll
    for(int offset = warpSize >> 1; offset > 0; offset >>= 1) 
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    if(laneId == 0) s_y[warpId] = val;
    __syncthreads();

    if(warpId == 0)
    {
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum) ? s_y[laneId] : 0.0f;
        for(int offset = warpSize >> 1; offset > 0; offset >>= 1) 
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if(laneId == 0) atomicAdd(d_y, val);
    }
}

__global__ void reduce_v4(float* d_x, float* d_y, const int N)
{
    __shared__ float s_y[32];

    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x & (warpSize - 1);

    float val = 0.0f;
    if(idx < N) 
    {
        float4 tmp_x = FLOAT4(d_x[idx]);
        val += tmp_x.x;
        val += tmp_x.y;
        val += tmp_x.z;
        val += tmp_x.w;
    }

    #pragma unroll
    for(int offset = warpSize >> 1; offset > 0; offset >>= 1) 
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    if(laneId == 0) s_y[warpId] = val;
    __syncthreads();

    if(warpId == 0)
    {
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum) ? s_y[laneId] : 0.0f;
        for(int offset = warpSize >> 1; offset > 0; offset >>= 1) 
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if(laneId == 0) atomicAdd(d_y, val);
    }
}