#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <iostream>

int main() {
    // 定义输入数据大小
    const int N = 10;
    
    // 在主机上创建并初始化输入数据
    thrust::host_vector<int> h_input(N);
    for(int i = 0; i < N; i++) {
        h_input[i] = i + 1;  // 1, 2, 3, ..., 10
    }
    
    // 将数据从主机复制到设备
    thrust::device_vector<int> d_input = h_input;
    
    // 在设备上创建输出向量
    thrust::device_vector<int> d_output(N);
    
    // 使用thrust::inclusive_scan计算前缀和
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());
    
    // 将结果从设备复制回主机
    thrust::host_vector<int> h_output = d_output;
    
    // 打印输入和输出
    std::cout << "输入数据: ";
    for(int i = 0; i < N; i++) {
        std::cout << h_input[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "前缀和结果: ";
    for(int i = 0; i < N; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}