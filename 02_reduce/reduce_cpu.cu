#include "error.cuh"
#include <stdio.h>

//编译的时候 -DUSE_DP
#ifdef USE_DP
    typedef double real;
#else 
    typedef float real;
#endif


const int NUM_REPEATS = 20;
void timing(const real *x, const int N);
real reduce(const real *x, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(real) * N;
    real *x = (real *)malloc(M);
    
    for(int i = 0; i < N; ++i) x[i] = 1.23;

    timing(x, N);

    free(x);
    return 0;
}

void timing(const real *x, const int N)
{
    real sum = 0;
    float mi = 99999, mx = 0, cnt = 0;
    for(int i = 0; i < NUM_REPEATS; ++i)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start); //记录的不是cpu执行到这行代码时的事件
                               //记录的时GPU时间，即GPU开始处理的时间

        sum = reduce(x, N);

        //先记录再同步是没问题的，正常是异步的，同步是为了后面计算时间差时，stop是正确的
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
        mi = min(mi, elapsed_time);
        mx = max(mx, elapsed_time);
        cnt += elapsed_time;
    }

    printf("sum = %f.\n", sum);
    printf("mi = %f mx = %f avg = %f\n", mi, mx, cnt / NUM_REPEATS);
}

real reduce(const real *x, const int N)
{
    real sum = 0.0;
    for(int i = 0; i < N; ++i) sum += x[i];
    return sum;
}