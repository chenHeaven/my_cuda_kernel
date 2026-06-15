#include <iostream>
#include <stdio.h>
#include <vector>
#include "../common.h"

float reduce_cpu(const std::vector<float>& data) {
    float sum = 0.0f;
    for (float val : data) {
        sum += val;
    }
    return sum;
}

__global__ void reduceSum(const float* inp, float* out, int n, int warpSize = 32) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    float sum = 0.0f;
    __shared__ float warpNum[32];
    for (int i = id; i < n; i += gridDim.x * blockDim.x) {
        sum += inp[i];
    }
    sum = warp_reduceSum(sum);
    if (lane == 0) {
        warpNum[warpId] = sum;
    }
    __syncthreads();
    if (warpId == 0) {
        sum = (threadIdx.x < blockDim.x / warpSize) ? warpNum[threadIdx.x] : 0.0f;
        sum = warp_reduceSum(sum);
    }
    if (threadIdx.x == 0) {
        out[blockIdx.x] = sum;
    }
}
void reduceSum_v1(const float* inp, float* out, int n, int blockdim) {
    int griddim = ceil_div(n, blockdim);
    reduceSum<<<griddim, blockdim>>>(inp, out, n);
    reduceSum<<<1, blockdim>>>(inp, out, n);
}


int main() {
    const int block_size = 1024;
    const int n = 1024 * 1024;
    std::vector<float> h_data(n, 1.0f);
    auto [cpu_result, duration] = benchmark_cpu(reduce_cpu, h_data);
    float* d_inp;
    float* d_out;
    float g_time;
    float gpu_result;

    int size = n * sizeof(float);
    cudaCheck(cudaMalloc(&d_inp, size));
    cudaCheck(cudaMalloc(&d_out, size));
    cudaCheck(cudaMemcpy(d_inp, h_data.data(), size, cudaMemcpyHostToDevice));
    g_time = benchmark_kernel(10, reduceSum_v1, d_inp, d_out, n, block_size);
    cudaCheck(cudaMemcpy(&gpu_result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    if (fabsf(gpu_result - cpu_result) > 1e-6f) {
        std::cout<<"结果不同"<<std::endl;
    }
    else {
       std::cout<<"结果相同"<<std::endl; 
    }
    std::cout<<"cpu运行时间"<<duration.count()<<"ms gpu运行时间"<<g_time<<"ms"<<std::endl; 
}