#include "error.cuh"
#include "reduce.cuh"
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 200;
const int N = 100000000;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;

void timing(real *d_x, const int method);
real reduce(real *d_x, const int method);

int main(int argc, char* argv[])
{
    real *h_x = (real *) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = 1.23;
    }
    real *d_x;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

    timing(d_x, atoi(argv[1]));

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

real reduce(real *d_x, const int method)
{
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int smem = sizeof(real) * BLOCK_SIZE;
    real result = 0.0;

    if(method < 3)
    {
        const int ymem = sizeof(real) * grid_size;
        real *h_y = (real *) malloc(ymem);

        real *d_y;
        CHECK(cudaMalloc(&d_y, ymem));
        
        if(method == 0) reduce_global<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);
        else if(method == 1) reduce_shared<<<grid_size, BLOCK_SIZE>>>(d_x, d_y, N);
        else reduce_dynamic<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);
        
        CHECK(cudaMemcpy(h_y, d_y, ymem, cudaMemcpyDeviceToHost));
        
        for (int n = 0; n < grid_size; ++n)
        {
            //printf("%d %f\n", n, h_y[n]);
            result += h_y[n];
        }
        
        CHECK(cudaFree(d_y));
        free(h_y);
    }
    else
    {
        real h_y[1] = {0};
        real *d_y;
        CHECK(cudaMalloc(&d_y, sizeof(real)));
        CHECK(cudaMemcpy(d_y, h_y, sizeof(real), cudaMemcpyHostToDevice));
        
        if(method == 3) reduce_atomic<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);
        else if(method == 4) reduce_syncwarp<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);
        else if(method == 5) reduce_shfl<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);
        else if(method == 6) reduce_cp<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);
        else if(method == 7) reduce_idle<<<10240, BLOCK_SIZE, smem>>>(d_x, d_y, N);
        else 
        {
            const int blockSize = 1024;
            int gridSize = (N + 4 * blockSize - 1) / (4 * blockSize);
            reduce_best<blockSize><<<gridSize, blockSize>>>(d_x, d_y, N);
        }

        CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d_y));

        result = h_y[0];
    }
    
    return result;
}

void timing(real *d_x, const int method)
{
    real sum = 0;

    float mi = 99999, mx = 0, cnt = 0;
    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        //CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
        
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(d_x, method);
        //printf("%d %f\n", repeat, sum);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
        if(repeat > 99)
        {
            mi = min(mi, elapsed_time);
            mx = max(mx, elapsed_time);
            cnt += elapsed_time;
        }
    }

    printf("sum = %f.\n", sum);
    printf("mi = %f mx = %f avg = %f\n", mi, mx, cnt / 100);
}