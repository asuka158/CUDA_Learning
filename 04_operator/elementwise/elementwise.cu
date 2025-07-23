#define CEIL(a, b) ((a + b - 1) / (b))
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

int N;

int block_size = 1024;
int grid_size = CEIL(N, block_size);

__global__ void elementwise_add(float* a, float* b, float* c, ,int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N) c[idx] = a[idx] + b[idx];
}

__global__ void elementwise_add_float4(float* a, float* b, float* c, int N)
{
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;

    if(idx < N)
    {
        float4 tmp_a = FLOAT4(a[idx]);
        float4 tmp_b = FLOAT4(b[idx]);
        float4 tmp_c;
        tmp_c.x = tmp_a.x + tmp_b.x;
        tmp_c.y = tmp_a.y + tmp_b.y;
        tmp_c.z = tmp_a.z + tmp_b.z;
        tmp_c.w = tmp_a.w + tmp_b.w;
        FLOAT4(c[idx]) = tmp_c;
    }
}

__global__ void sigmoid(float* x, float* y, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N) y[idx] = 1.0f / (1.0f + expf(-x[idx]));
}

__global__ void sigmoid_float4(float* x, float* y, int N)
{
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if(idx < N)
    {
        float4 tmp_x = FLOAT4(x[idx]);
        float4 tmp_y;
        tmp_y.x = 1.0f / (1.0f + expf(-tmp_x.x));
        tmp_y.y = 1.0f / (1.0f + expf(-tmp_x.y));
        tmp_y.z = 1.0f / (1.0f + expf(-tmp_x.z));
        tmp_y.w = 1.0f / (1.0f + expf(-tmp_x.w));
        FLOAT4(y[idx]) = tmp_y;
    }
}

__global__ void relu(float* x, float* y, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N) y[idx] = fmaxf(0.0f, x[idx]);
}

__global__ void relu_float4(float* x, float* y, int N)
{
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if(idx < N)
    {
        float4 tmp_x = FLOAT4(x[idx]);
        float4 tmp_y;
        tmp_y.x = fmaxf(0.0f, tmp_x.x);
        tmp_y.y = fmaxf(0.0f, tmp_x.y);
        tmp_y.z = fmaxf(0.0f, tmp_x.z);
        tmp_y.w = fmaxf(0.0f, tmp_x.w);
        FLOAT4(y[idx]) = tmp_y;
    }
}