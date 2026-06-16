//图像颜色反转，image输入时A、B、G、R，A时透明度，A不变，其余颜色取反，或者是减去255

#include <cuda_runtime.h>
#include "../common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

__global__ void image_inversion_naive(unsigned int* image, int width, int height) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int pixel_count = width * height;

    if (id >= pixel_count) return;
    image[id] ^= 0x00FFFFFFu;
}

__global__ void image_inversion_v1(unsigned int* image, int width, int height) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int pixel_count = width * height;
    int vec_count = pixel_count / 4;

    if (id < vec_count) {
        uint4* abgrPtr4 = reinterpret_cast<uint4*>(image);
        uint4 abgr = abgrPtr4[id];
        abgrPtr4[id] = make_uint4(abgr.x ^ 0x00FFFFFFu,
                                  abgr.y ^ 0x00FFFFFFu,
                                  abgr.z ^ 0x00FFFFFFu,
                                  abgr.w ^ 0x00FFFFFFu);
    }
    if (id < (pixel_count & 3)) {
        int tail_idx = vec_count * 4 + id;
        image[tail_idx] ^= 0x00FFFFFFu;
    }
}

std::vector<int> generateSizes() { return {1024, 2048, 4096}; }

int main() {
    int device_id = 0;
    cudaCheck(cudaSetDevice(device_id));

    constexpr int repeat_time = 10;
    constexpr int block_size = 256;
    std::ofstream csv_file("image_inversion_benchmark.csv");
    csv_file << "Width,Height,Naive_TIME_MS,V1_TIME_MS,Naive_GBPS,V1_GBPS,Matched" << std::endl;

    for (int n : generateSizes()) {
        int width = n;
        int height = n;
        size_t pixel_count = static_cast<size_t>(width) * height;
        size_t bytes = pixel_count * sizeof(unsigned int);

        std::cout << "Testing size: " << width << "x" << height << std::endl;

        std::vector<unsigned int> h_input(pixel_count);
        std::vector<unsigned int> h_naive(pixel_count);
        std::vector<unsigned int> h_v1(pixel_count);

        for (size_t i = 0; i < pixel_count; ++i) {
            unsigned int alpha = 0xFF000000u;
            unsigned int rgb = static_cast<unsigned int>(i * 2654435761u) & 0x00FFFFFFu;
            h_input[i] = alpha | rgb;
        }

        unsigned int *d_naive, *d_v1;
        cudaCheck(cudaMalloc(&d_naive, bytes));
        cudaCheck(cudaMalloc(&d_v1, bytes));
        cudaCheck(cudaMemcpy(d_naive, h_input.data(), bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_v1, h_input.data(), bytes, cudaMemcpyHostToDevice));

        int naive_grid = CEIL_DIV(pixel_count, block_size);
        float naive_time = benchmark_kernel(repeat_time, [&]() {
            image_inversion_naive<<<naive_grid, block_size>>>(d_naive, width, height);
            cudaCheck(cudaGetLastError());
        });

        int v1_grid = CEIL_DIV(CEIL_DIV(pixel_count, 4), block_size);
        float v1_time = benchmark_kernel(repeat_time, [&]() {
            image_inversion_v1<<<v1_grid, block_size>>>(d_v1, width, height);
            cudaCheck(cudaGetLastError());
        });

        cudaCheck(cudaMemcpy(h_naive.data(), d_naive, bytes, cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(h_v1.data(), d_v1, bytes, cudaMemcpyDeviceToHost));

        bool matched = std::memcmp(h_naive.data(), h_v1.data(), bytes) == 0;
        float data_gb = static_cast<float>(bytes) / 1e9f;
        float naive_gbps = data_gb / (naive_time / 1000.0f);
        float v1_gbps = data_gb / (v1_time / 1000.0f);

        std::cout << "Naive avg time: " << naive_time
                  << " ms, V1 avg time: " << v1_time
                  << " ms, matched: " << matched << std::endl;

        csv_file << width << "," << height << ","
                 << naive_time << "," << v1_time << ","
                 << naive_gbps << "," << v1_gbps << ","
                 << (matched ? "1" : "0") << std::endl;

        cudaCheck(cudaFree(d_naive));
        cudaCheck(cudaFree(d_v1));
    }

    csv_file.close();
    std::cout << "Benchmark completed. Results saved to 'image_inversion_benchmark.csv'" << std::endl;
    return 0;
}
