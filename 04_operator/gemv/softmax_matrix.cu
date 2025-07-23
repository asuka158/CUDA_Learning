#define CEIL(a, b) ((a + b - 1) / (b))
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

__global__ void softmax_kernel_v1(float* input, float* output, int M, int N)
{
    __shared__ float s_max_val;
    __shared__ float s_sum;
    int laneId = threadIdx.x & (warpSize - 1);

    int row = blockIdx.x;
    if(row >= M) return ;

    int cnt = CEIL(N, warpSize);

    float max_val = -FLT_MAX;
    for(int i = 0; i < cnt; ++i)
    {
        int col = i * warpSize + laneId;
        max_val = (col < N) ? fmaxf(max_val, input[row * N + col]) : max_val;
    }

    for(int offset = warpSize >> 1; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    
    if(laneId == 0) s_max_val = max_val;

    float sum = 0.0f;
    for(int i = 0; i < cnt; ++i)
    {
        int col = i * warpSize + laneId;
        sum += (col < N) ? expf(input[row * N + col] - s_max_val) : 0.0f;
    }
    for(int offset = warpSize >> 1; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if(laneId == 0) s_sum = sum;

    for(int i = 0; i < cnt; ++i)
    {
        int col = i * warpSize + laneId;
        if(col < N) output[row * N + col] = expf(input[row * N + col] - s_max_val) / s_sum;
    }
}

__global__ void softmax_kernel_v2(float* input, float* output, int M, int N)
{
    int laneId = threadIdx.x & (warpSize - 1);

    int row = blockIdx.x;
    if(row >= M) return ;

    int cnt = CEIL(N, warpSize);

    float max_val = -FLT_MAX;
    for(int i = 0; i < cnt; ++i)
    {
        int col = i * warpSize + laneId;
        max_val = (col < N) ? fmaxf(max_val, input[row * N + col]) : max_val;
    }

    for(int offset = warpSize >> 1; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));

    float sum = 0.0f;
    for(int i = 0; i < cnt; ++i)
    {
        int col = i * warpSize + laneId;
        sum += (col < N) ? expf(input[row * N + col] - s_max_val) : 0.0f;
    }
    for(int offset = warpSize >> 1; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);

    for(int i = 0; i < cnt; ++i)
    {
        int col = i * warpSize + laneId;
        if(col < N) output[row * N + col] = expf(input[row * N + col] - s_max_val) / s_sum;
    }
}