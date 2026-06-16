#include <cuda_runtime.h>
#include "../common.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#define blockSize 256
#define elePerT 4
#define totalPerT (blockSize * elePerT)
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

__global__ void reverse_array_naive(float* input, int N) {
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < N / 2) {
        float a = input[tid];
        float b = input[N - tid - 1];

        input[tid] = b;
        input[N - tid - 1] = a;
    }
}

__global__ void reverse_array_v1(float* input, int N) {

    __shared__ float inputShared[totalPerT * 2];
    int half = N / 2;
    for (int i = 0; i < elePerT; ++i) {
        int localIdx = i * blockSize + threadIdx.x;
        int globalIdx = blockIdx.x * totalPerT + localIdx;
        if (globalIdx < half) {
            inputShared[localIdx] = input[globalIdx];
            inputShared[totalPerT + localIdx] = input[N - 1 - globalIdx];
        }
    }
    __syncthreads();
    for (int i = 0; i < elePerT; ++i) {
        int localIdx = i * blockSize + threadIdx.x;
        int globalIdx = blockIdx.x * totalPerT + localIdx;
        if (globalIdx < half) {
            input[globalIdx] = inputShared[totalPerT + localIdx];
            input[N - 1 - globalIdx] = inputShared[localIdx];
        }
    }
}

std::vector<int> generateSizes() { return {1 << 20, 1 << 22, 1 << 24}; }

int main() {
    int device_id = 0;
    cudaCheck(cudaSetDevice(device_id));

    constexpr int repeat_time = 10;
    std::ofstream csv_file("reverse_array_benchmark.csv");
    csv_file << "Size,Naive_TIME_MS,V1_TIME_MS,Naive_GBPS,V1_GBPS,Matched" << std::endl;

    for (int N : generateSizes()) {
        size_t bytes = static_cast<size_t>(N) * sizeof(float);
        std::cout << "Testing size: " << N << std::endl;

        std::vector<float> h_input(N);
        std::vector<float> h_expected(N);
        std::vector<float> h_naive(N);
        std::vector<float> h_v1(N);

        for (int i = 0; i < N; ++i) {
            h_input[i] = static_cast<float>((i * 17) % 1000) * 0.001f;
        }
        h_expected = h_input;
        std::reverse(h_expected.begin(), h_expected.end());

        float *d_naive, *d_v1;
        cudaCheck(cudaMalloc(&d_naive, bytes));
        cudaCheck(cudaMalloc(&d_v1, bytes));
        cudaCheck(cudaMemcpy(d_naive, h_input.data(), bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_v1, h_input.data(), bytes, cudaMemcpyHostToDevice));

        int naive_grid = CEIL_DIV(N / 2, blockSize);
        float naive_time = benchmark_kernel(repeat_time, [&]() {
            reverse_array_naive<<<naive_grid, blockSize>>>(d_naive, N);
            cudaCheck(cudaGetLastError());
        });

        int v1_grid = CEIL_DIV(N / 2, totalPerT);
        float v1_time = benchmark_kernel(repeat_time, [&]() {
            reverse_array_v1<<<v1_grid, blockSize>>>(d_v1, N);
            cudaCheck(cudaGetLastError());
        });

        cudaCheck(cudaMemcpy(d_naive, h_input.data(), bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_v1, h_input.data(), bytes, cudaMemcpyHostToDevice));
        reverse_array_naive<<<naive_grid, blockSize>>>(d_naive, N);
        cudaCheck(cudaGetLastError());
        reverse_array_v1<<<v1_grid, blockSize>>>(d_v1, N);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        cudaCheck(cudaMemcpy(h_naive.data(), d_naive, bytes, cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(h_v1.data(), d_v1, bytes, cudaMemcpyDeviceToHost));

        bool matched = compare_result(h_expected.data(), h_naive.data(), N, 1e-6f) &&
                       compare_result(h_expected.data(), h_v1.data(), N, 1e-6f);
        float data_gb = static_cast<float>(2.0 * bytes) / 1e9f;
        float naive_gbps = data_gb / (naive_time / 1000.0f);
        float v1_gbps = data_gb / (v1_time / 1000.0f);

        std::cout << "Naive avg time: " << naive_time
                  << " ms, V1 avg time: " << v1_time
                  << " ms, matched: " << matched << std::endl;

        csv_file << N << ","
                 << naive_time << "," << v1_time << ","
                 << naive_gbps << "," << v1_gbps << ","
                 << (matched ? "1" : "0") << std::endl;

        cudaCheck(cudaFree(d_naive));
        cudaCheck(cudaFree(d_v1));
    }

    csv_file.close();
    std::cout << "Benchmark completed. Results saved to 'reverse_array_benchmark.csv'" << std::endl;
    return 0;
}
