//矩阵加法

#include <cuda_runtime.h>
#include "../common.h"
#include <fstream>
#include <iostream>
#include <vector>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

__global__ void matrix_add_naive(const float* A, const float* B, float* C, int N) 
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if(x < N && y < N){
        C[y * N + x] = A[y * N + x] + B[y * N + x];
    }
}

__global__ void matrix_add_v1(const float* A, const float* B, float* C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_element = N * N;
    int vec_count = total_element / 4;
    if (idx < vec_count) {
        float4 AA = reinterpret_cast<const float4*>(A)[idx];
        float4 BB = reinterpret_cast<const float4*>(B)[idx];
        reinterpret_cast<float4*>(C)[idx] = make_float4(AA.x + BB.x, AA.y + BB.y, AA.z + BB.z, AA.w + BB.w);
    }
    if (idx < (total_element & 3)) {
        int tail_idx = vec_count * 4 + idx;
        C[tail_idx] = A[tail_idx] + B[tail_idx];
    }
}

std::vector<int> generateSizes() { return {1024, 2048, 4096}; }

int main() {
    int device_id = 0;
    cudaCheck(cudaSetDevice(device_id));

    constexpr int repeat_time = 10;
    constexpr int block_size = 256;
    std::ofstream csv_file("matrix_addition_benchmark.csv");
    csv_file << "Size,Naive_TIME_MS,V1_TIME_MS,Naive_GBPS,V1_GBPS,Matched" << std::endl;

    for (int N : generateSizes()) {
        size_t elem_count = static_cast<size_t>(N) * N;
        size_t bytes = elem_count * sizeof(float);

        std::cout << "Testing size: " << N << "x" << N << std::endl;

        std::vector<float> h_A(elem_count);
        std::vector<float> h_B(elem_count);
        std::vector<float> h_naive(elem_count);
        std::vector<float> h_v1(elem_count);

        for (size_t i = 0; i < elem_count; ++i) {
            h_A[i] = static_cast<float>(i % 1000) * 0.001f;
            h_B[i] = static_cast<float>((i * 7) % 1000) * 0.002f;
        }

        float *d_A, *d_B, *d_naive, *d_v1;
        cudaCheck(cudaMalloc(&d_A, bytes));
        cudaCheck(cudaMalloc(&d_B, bytes));
        cudaCheck(cudaMalloc(&d_naive, bytes));
        cudaCheck(cudaMalloc(&d_v1, bytes));

        cudaCheck(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemset(d_naive, 0, bytes));
        cudaCheck(cudaMemset(d_v1, 0, bytes));

        dim3 naive_block(32, 8);
        dim3 naive_grid(CEIL_DIV(N, naive_block.x), CEIL_DIV(N, naive_block.y));
        float naive_time = benchmark_kernel(repeat_time, [&]() {
            matrix_add_naive<<<naive_grid, naive_block>>>(d_A, d_B, d_naive, N);
            cudaCheck(cudaGetLastError());
        });

        int v1_grid = CEIL_DIV(elem_count / 4, block_size);
        float v1_time = benchmark_kernel(repeat_time, [&]() {
            matrix_add_v1<<<v1_grid, block_size>>>(d_A, d_B, d_v1, N);
            cudaCheck(cudaGetLastError());
        });

        cudaCheck(cudaMemcpy(h_naive.data(), d_naive, bytes, cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(h_v1.data(), d_v1, bytes, cudaMemcpyDeviceToHost));

        bool matched = compare_result(h_naive.data(), h_v1.data(), elem_count, 1e-6f);
        float data_gb = static_cast<float>(3.0 * bytes) / 1e9f;
        float naive_gbps = data_gb / (naive_time / 1000.0f);
        float v1_gbps = data_gb / (v1_time / 1000.0f);

        std::cout << "Naive avg time: " << naive_time
                  << " ms, V1 avg time: " << v1_time
                  << " ms, matched: " << matched << std::endl;

        csv_file << N << ","
                 << naive_time << "," << v1_time << ","
                 << naive_gbps << "," << v1_gbps << ","
                 << (matched ? "1" : "0") << std::endl;

        cudaCheck(cudaFree(d_A));
        cudaCheck(cudaFree(d_B));
        cudaCheck(cudaFree(d_naive));
        cudaCheck(cudaFree(d_v1));
    }

    csv_file.close();
    std::cout << "Benchmark completed. Results saved to 'matrix_addition_benchmark.csv'" << std::endl;
    return 0;
}
