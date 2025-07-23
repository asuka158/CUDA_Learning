__global__ void transpose(float* input, float* output, int M, int N)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < M && col < N) output[col * M + row] = input[row * N + col];
}

__global__ void transpose_v1(float* input, float* output, int M, int N)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < N && col < M) output[row * M + col] = __ldg(&input[col * N + row]);
}

dim3 block(32, 32);
dim3 grid(CEIL(N, 32), CEIL(M, 32));

template <const int BLOCK_SIZE>
__global__ void transpose_v2(float* input, float* output, int M, int N)
{
    __shared__ float s_mem[BLOCK_SIZE][BLOCK_SIZE + 1];
    
    int bx = blockIdx.x * BLOCK_SIZE;
    int by = blockIdx.y * BLOCK_SIZE;
    int x1 = bx + threadIdx.x;
    int y1 = by + threadIdx.y;

    if(x1 < N && y1 < M) s_mem[threadIdx.y][threadIdx.x] = input[y1 * N + x1];
    __syncthreads();

    int x2 = by + threadIdx.x;
    int y2 = bx + threadIdx.y;
    if(x2 < M && y2 < N) output[y2 * M + x2] = s_mem[threadIdx.x][threadIdx.y];
}