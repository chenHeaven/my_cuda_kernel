//矩阵转置

#include <cuda_runtime.h>
#include "../common.h"
#include <fstream>
#include <iostream>
#include <vector>

#define TILE_DIM 32
#define BLOCK_ROWS 2
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

__global__ void transpose_naive(const float* input, float* output, int rows, int cols) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx < cols && ty < rows) {
        output[tx * rows + ty] = input[ty * cols + tx];
    }
}

__global__ void transpose_v1(const float* input, float* output, int rows, int cols) {
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; //防止bank conflict
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        int y = yIndex + i;
        if (y < rows && xIndex < cols) {
            tile[threadIdx.y + i][threadIdx.x] = input[y * cols + xIndex];
        }
    }

    __syncthreads();

    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        int y = yIndex + i;
        if (y < cols && xIndex < rows) {
            output[y * rows + xIndex] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}


std::vector<int> generateSizes() { return {1024, 2048, 4096}; }

int main() {
    int device_id = 0;
    cudaCheck(cudaSetDevice(device_id));

    constexpr int repeat_time = 10;
    std::ofstream csv_file("transpose_benchmark.csv");
    csv_file << "Rows,Cols,Naive_TIME_MS,V1_TIME_MS,Naive_GBPS,V1_GBPS,Matched" << std::endl;

    for (int n : generateSizes()) {
        int rows = n;
        int cols = n;
        size_t elem_count = static_cast<size_t>(rows) * cols;
        size_t bytes = elem_count * sizeof(float);

        std::cout << "Testing size: " << rows << "x" << cols << std::endl;

        float* h_input = static_cast<float*>(malloc(bytes));
        float* h_naive = static_cast<float*>(malloc(bytes));
        float* h_v1 = static_cast<float*>(malloc(bytes));

        for (size_t i = 0; i < elem_count; ++i) {
            h_input[i] = static_cast<float>(i % 1000) * 0.001f;
        }

        float *d_input, *d_naive, *d_v1;
        cudaCheck(cudaMalloc(&d_input, bytes));
        cudaCheck(cudaMalloc(&d_naive, bytes));
        cudaCheck(cudaMalloc(&d_v1, bytes));

        cudaCheck(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemset(d_naive, 0, bytes));
        cudaCheck(cudaMemset(d_v1, 0, bytes));

        dim3 naive_block(32, 8);
        dim3 naive_grid(CEIL_DIV(cols, naive_block.x), CEIL_DIV(rows, naive_block.y));
        float naive_time = benchmark_kernel(repeat_time, [&]() {
            transpose_naive<<<naive_grid, naive_block>>>(d_input, d_naive, rows, cols);
            cudaCheck(cudaGetLastError());
        });

        dim3 v1_block(TILE_DIM, BLOCK_ROWS);
        dim3 v1_grid(CEIL_DIV(cols, TILE_DIM), CEIL_DIV(rows, TILE_DIM));
        float v1_time = benchmark_kernel(repeat_time, [&]() {
            transpose_v1<<<v1_grid, v1_block>>>(d_input, d_v1, rows, cols);
            cudaCheck(cudaGetLastError());
        });

        cudaCheck(cudaMemcpy(h_naive, d_naive, bytes, cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(h_v1, d_v1, bytes, cudaMemcpyDeviceToHost));

        bool matched = compare_result(h_naive, h_v1, elem_count, 1e-6f);
        float data_gb = static_cast<float>(2.0 * bytes) / 1e9f;
        float naive_gbps = data_gb / (naive_time / 1000.0f);
        float v1_gbps = data_gb / (v1_time / 1000.0f);

        std::cout << "Naive avg time: " << naive_time
                  << " ms, V1 avg time: " << v1_time
                  << " ms, matched: " << matched << std::endl;

        csv_file << rows << "," << cols << ","
                 << naive_time << "," << v1_time << ","
                 << naive_gbps << "," << v1_gbps << ","
                 << (matched ? "1" : "0") << std::endl;

        cudaCheck(cudaFree(d_input));
        cudaCheck(cudaFree(d_naive));
        cudaCheck(cudaFree(d_v1));
        free(h_input);
        free(h_naive);
        free(h_v1);
    }

    csv_file.close();
    std::cout << "Benchmark completed. Results saved to 'transpose_benchmark.csv'" << std::endl;
    return 0;
}
