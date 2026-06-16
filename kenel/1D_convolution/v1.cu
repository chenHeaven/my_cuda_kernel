#include <cuda_runtime.h>
#include "../common.h"
#include <fstream>
#include <iostream>
#include <vector>

#define blockSize 256
#define perElementThread 4
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

__constant__ float kernelData[2048];

__global__ void conv1d_naive(const float* __restrict__ input,
                             const float* __restrict__ kernel,
                             float* __restrict__ output,
                             int inputSize,
                             int kernelSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int outputSize = inputSize - kernelSize + 1;
    if (idx >= outputSize) return;

    float sum = 0.0f;
    for (int j = 0; j < kernelSize; ++j) {
        sum += input[idx + j] * kernel[j];
    }
    output[idx] = sum;
}

__global__ void conv1d_v1(const float* __restrict__ input, float* __restrict__ output, int inputSize, int kernelSize) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int elementPerBolck = blockSize * perElementThread;
    int blockBegin = bid * elementPerBolck;
    int outputSize = inputSize - kernelSize + 1;

    int blockOutSize = min(elementPerBolck, outputSize - blockBegin);
    int blockInputSize = min(blockOutSize + kernelSize - 1, inputSize - blockBegin);


    extern __shared__ float inputShared[];
    int float4Num = blockInputSize >> 2;
    if (blockBegin >= outputSize) return;
    for (int i = tid; i < float4Num; i += blockSize) {
        reinterpret_cast<float4*>(inputShared)[i] = reinterpret_cast<const float4*>(input + blockBegin)[i];
    }
    if (tid < (blockInputSize & 3)) {
        int tail = blockInputSize & ~3;
        inputShared[tail + tid] = input[tail + blockBegin + tid];
    }
    __syncthreads();

    float tile[perElementThread] = {0.0f};
    int kernelFloat4 = kernelSize >> 2;
    int threadOutputStart = tid * perElementThread;
    // if (threadOutputStart >= blockOutSize) return;
    for (int j = 0; j < kernelFloat4; ++j) {
        float4 kernelDataFloat4 = reinterpret_cast<const float4*>(kernelData)[j];
        for (int i = 0; i < perElementThread; ++i) {
            if (threadOutputStart + i >= blockOutSize) continue;
            tile[i] += kernelDataFloat4.x * inputShared[tid * perElementThread + i + j * 4 + 0];
            tile[i] += kernelDataFloat4.y * inputShared[tid * perElementThread + i + j * 4 + 1];
            tile[i] += kernelDataFloat4.z * inputShared[tid * perElementThread + i + j * 4 + 2];
            tile[i] += kernelDataFloat4.w * inputShared[tid * perElementThread + i + j * 4 + 3];
        }
    }
    for (int j = (kernelSize & ~3); j < kernelSize; ++j) {
        for (int i = 0; i < perElementThread; ++i) {
            if (threadOutputStart + i >= blockOutSize) continue;
            tile[i] += kernelData[j] * inputShared[tid * perElementThread + i + j];
        }
    }
    for (int i = 0; i < perElementThread; ++i) {
        if ((threadOutputStart + blockBegin + i) < outputSize) {
            output[i + threadOutputStart + blockBegin] = tile[i];
        }
    }
}

std::vector<int> generateSizes() { return {1 << 20, 1 << 22, 1 << 24}; }

int main() {
    int device_id = 0;
    cudaCheck(cudaSetDevice(device_id));

    constexpr int repeat_time = 10;
    constexpr int kernelSize = 63;
    std::ofstream csv_file("conv1d_benchmark.csv");
    csv_file << "InputSize,KernelSize,Naive_TIME_MS,V1_TIME_MS,Naive_GBPS,V1_GBPS,Matched" << std::endl;

    for (int inputSize : generateSizes()) {
        int outputSize = inputSize - kernelSize + 1;
        size_t inputBytes = static_cast<size_t>(inputSize) * sizeof(float);
        size_t kernelBytes = static_cast<size_t>(kernelSize) * sizeof(float);
        size_t outputBytes = static_cast<size_t>(outputSize) * sizeof(float);

        std::cout << "Testing inputSize: " << inputSize
                  << ", kernelSize: " << kernelSize << std::endl;

        std::vector<float> h_input(inputSize);
        std::vector<float> h_kernel(kernelSize);
        std::vector<float> h_naive(outputSize);
        std::vector<float> h_v1(outputSize);

        for (int i = 0; i < inputSize; ++i) {
            h_input[i] = static_cast<float>((i * 17) % 1000) * 0.001f;
        }
        for (int i = 0; i < kernelSize; ++i) {
            h_kernel[i] = static_cast<float>((i * 13) % 31) * 0.01f;
        }

        float *d_input, *d_kernel, *d_naive, *d_v1;
        cudaCheck(cudaMalloc(&d_input, inputBytes));
        cudaCheck(cudaMalloc(&d_kernel, kernelBytes));
        cudaCheck(cudaMalloc(&d_naive, outputBytes));
        cudaCheck(cudaMalloc(&d_v1, outputBytes));

        cudaCheck(cudaMemcpy(d_input, h_input.data(), inputBytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_kernel, h_kernel.data(), kernelBytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemset(d_naive, 0, outputBytes));
        cudaCheck(cudaMemset(d_v1, 0, outputBytes));
        cudaCheck(cudaMemcpyToSymbol(kernelData, d_kernel, kernelBytes, 0, cudaMemcpyDeviceToDevice));

        int naive_grid = CEIL_DIV(outputSize, blockSize);
        float naive_time = benchmark_kernel(repeat_time, [&]() {
            conv1d_naive<<<naive_grid, blockSize>>>(d_input, d_kernel, d_naive, inputSize, kernelSize);
            cudaCheck(cudaGetLastError());
        });

        int elementPerBlock = blockSize * perElementThread;
        int v1_grid = CEIL_DIV(outputSize, elementPerBlock);
        int sharedBytes = (elementPerBlock + kernelSize - 1) * sizeof(float);
        float v1_time = benchmark_kernel(repeat_time, [&]() {
            conv1d_v1<<<v1_grid, blockSize, sharedBytes>>>(d_input, d_v1, inputSize, kernelSize);
            cudaCheck(cudaGetLastError());
        });

        cudaCheck(cudaMemcpy(h_naive.data(), d_naive, outputBytes, cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(h_v1.data(), d_v1, outputBytes, cudaMemcpyDeviceToHost));

        bool matched = compare_result(h_naive.data(), h_v1.data(), outputSize, 1e-4f);
        float data_gb = static_cast<float>(inputBytes + kernelBytes + outputBytes) / 1e9f;
        float naive_gbps = data_gb / (naive_time / 1000.0f);
        float v1_gbps = data_gb / (v1_time / 1000.0f);

        std::cout << "Naive avg time: " << naive_time
                  << " ms, V1 avg time: " << v1_time
                  << " ms, matched: " << matched << std::endl;

        csv_file << inputSize << "," << kernelSize << ","
                 << naive_time << "," << v1_time << ","
                 << naive_gbps << "," << v1_gbps << ","
                 << (matched ? "1" : "0") << std::endl;

        cudaCheck(cudaFree(d_input));
        cudaCheck(cudaFree(d_kernel));
        cudaCheck(cudaFree(d_naive));
        cudaCheck(cudaFree(d_v1));
    }

    csv_file.close();
    std::cout << "Benchmark completed. Results saved to 'conv1d_benchmark.csv'" << std::endl;
    return 0;
}
