// 待办
// 前缀和不像reduce一样简单，它存在明显的数据依赖，不能简单的并行计算
// 下面代码是ai生成的，待分析
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 核函数：上行扫描（reduce阶段）
__global__ void scan_up_kernel(int *d_data, int *d_sums, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int offset = 1;
    
    int ai = tid;
    int bi = tid + (n/2);
    
    // 加载数据到共享内存
    temp[ai] = d_data[ai];
    temp[bi] = d_data[bi];
    
    // 上行扫描
    for (int stride = n/2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            int idx = offset * (2*tid + 1) - 1;
            temp[idx + offset] += temp[idx];
        }
        offset <<= 1;
    }
    
    // 保存最后一个元素（总和）到全局内存
    if (tid == 0) {
        d_sums[blockIdx.x] = temp[n-1];
        temp[n-1] = 0; // 为下行扫描清零
    }
    
    // 下行扫描
    for (int stride = 1; stride < n; stride <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < stride) {
            int idx = offset * (2*tid + 1) - 1;
            int t = temp[idx];
            temp[idx] = temp[idx + offset];
            temp[idx + offset] += t;
        }
    }
    __syncthreads();
    
    // 写回结果
    d_data[ai] = temp[ai];
    d_data[bi] = temp[bi];
}

// 核函数：添加总和
__global__ void add_block_sums(int *d_data, int *d_sums, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    if (bid > 0) {
        int offset = d_sums[bid-1];
        d_data[tid] += offset;
        d_data[tid + (n/2)] += offset;
    }
}

// 主机函数：计算前缀和
void prefix_sum(int *h_data, int *h_result, int n) {
    int *d_data, *d_sums;
    int block_size = 512; // 假设每个块处理1024个元素（2 * 512线程）
    int num_blocks = (n + block_size*2 - 1) / (block_size*2);
    
    // 分配设备内存
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMalloc(&d_sums, num_blocks * sizeof(int));
    
    // 拷贝数据到设备
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // 调用上行扫描核函数
    scan_up_kernel<<<num_blocks, block_size, 2*block_size*sizeof(int)>>>(d_data, d_sums, 2*block_size);
    
    // 如果数据大于单个块能处理的大小，递归计算块的总和
    if (num_blocks > 1) {
        int *h_sums = (int*)malloc(num_blocks * sizeof(int));
        cudaMemcpy(h_sums, d_sums, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
        
        int *h_sums_scan = (int*)malloc(num_blocks * sizeof(int));
        prefix_sum(h_sums, h_sums_scan, num_blocks);
        
        cudaMemcpy(d_sums, h_sums_scan, num_blocks * sizeof(int), cudaMemcpyHostToDevice);
        free(h_sums);
        free(h_sums_scan);
        
        // 添加块的总和
        add_block_sums<<<num_blocks, block_size>>>(d_data, d_sums, 2*block_size);
    }
    
    // 拷贝结果回主机
    cudaMemcpy(h_result, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // 释放设备内存
    cudaFree(d_data);
    cudaFree(d_sums);
}

int main() {
    const int n = 1024;
    int h_data[n];
    int h_result[n];
    
    // 初始化数据
    for (int i = 0; i < n; i++) {
        h_data[i] = 1; // 或者任何其他值
    }
    
    // 计算前缀和
    prefix_sum(h_data, h_result, n);
    
    // 打印结果（前20个）
    for (int i = 0; i < 20; i++) {
        printf("%d ", h_result[i]);
    }
    printf("\n");
    
    return 0;
}