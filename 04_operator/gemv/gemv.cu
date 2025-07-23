// M = 1024
// N = 32
// block数量和行数相同: grid_size = M
// 每个block里一个warp: block_size = 32
#define CEIL(a, b) ((a + b - 1) / (b))
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

__global__ void sgemv(float* A, float* X, float* Y, int M, int K)
{
    int laneId = threadIdx.x & (warpSize - 1);
    int row = blockIdx.x;
    if(row >= M) return ;

    float res = 0.0f;
    int cnt = CEIL(K, warpSize);

    for(int i = 0; i < cnt; ++i)
    {
        int col = i * warpSize + laneId;
        res += (col < K) ? A[row * K + col] * X[col] : 0.0f;
    }

    for(int offset = warpSize >> 1; offset > 0; offset >>= 1) 
        res += __shfl_down_sync(0xFFFFFFFF, res, offset);

    if(laneId == 0) Y[row] = res;
}