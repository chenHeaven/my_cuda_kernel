#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "../common.h"

namespace cg = cooperative_groups;

constexpr int kThreadsPerBlock = 512;
constexpr int kMaxBlocks = 1024;

__device__ __forceinline__ void warp_max_sum(float& sum_self, float& max_self) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float sum_other = __shfl_down_sync(0xFFFFFFFF, sum_self, offset);
        float max_other = __shfl_down_sync(0xFFFFFFFF, max_self, offset);
        float max_temp = fmaxf(max_other, max_self);
        sum_self = sum_self * __expf(max_self - max_temp) + sum_other * __expf(max_other - max_temp);
        max_self = max_temp;
    }
}

__device__ float g_block_sum[kMaxBlocks];
__device__ float g_block_max[kMaxBlocks];

template <int threadsPerBlock>
__global__ void softmax_kernel(const float* input, float* output, int N) {
    cg::grid_group grid = cg::this_grid();

    int tid = threadIdx.x;
    int idx = tid + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x;

    float max_self = -FLT_MAX;
    float sum_self = 0.0f;
    for (int i = idx; i < N; i += stride) {
        float sum_other = input[i];
        float max_temp = fmaxf(max_self, sum_other);
        sum_self = sum_self * __expf(max_self - max_temp) + __expf(sum_other - max_temp);
        max_self = max_temp;
    }

    warp_max_sum(sum_self, max_self);

    int lane_id = tid % 32;
    int warp_id = tid / 32;
    int warpPerBlock = threadsPerBlock / 32;

    __shared__ float sum_shared[(threadsPerBlock / 32) * 2];
    if (lane_id == 0) {
        sum_shared[warp_id] = sum_self;
        sum_shared[(threadsPerBlock / 32) + warp_id] = max_self;
    }
    __syncthreads();

    if (warp_id == 0) {
        float sum = (tid < warpPerBlock) ? sum_shared[tid] : 0.0f;
        float max = (tid < warpPerBlock) ? sum_shared[warpPerBlock + tid] : -FLT_MAX;

        warp_max_sum(sum, max);

        if (tid == 0) {
            g_block_sum[blockIdx.x] = sum;
            g_block_max[blockIdx.x] = max;
        }
    }
    grid.sync();
    __shared__ float final_sum, final_max;
    if (warp_id == 0) {
        float b_sum = 0.0f;
        float b_max = -FLT_MAX;
        for (int i = tid; i < gridDim.x; i += 32) {
            float sum_other = g_block_sum[i];
            float max_other = g_block_max[i];
            float max_temp = fmaxf(b_max, max_other);
            b_sum = b_sum * __expf(b_max - max_temp) + sum_other * __expf(max_other - max_temp);
            b_max = max_temp;
        }
        warp_max_sum(b_sum, b_max);
        if (tid == 0) {
            final_sum = b_sum;
            final_max = b_max;
        }
    }
    __syncthreads();
    for (int i = idx; i < N; i += stride) {
        output[i] = __expf(input[i] - final_max) / final_sum;
    }
}

void softmax_gpu(const float* input, float* output, int N) {
    if (N <= 0) {
        return;
    }

    int device = 0;
    cudaCheck(cudaGetDevice(&device));

    int cooperative_launch = 0;
    cudaCheck(cudaDeviceGetAttribute(
        &cooperative_launch, cudaDevAttrCooperativeLaunch, device));
    if (!cooperative_launch) {
        std::cerr << "Current GPU does not support cooperative kernel launch."
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int sm_count = 0;
    cudaCheck(cudaDeviceGetAttribute(
        &sm_count, cudaDevAttrMultiProcessorCount, device));

    int blocks_per_sm = 0;
    cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, softmax_kernel<kThreadsPerBlock>,
        kThreadsPerBlock, 0));

    const int required_blocks = ceil_div(N, kThreadsPerBlock);
    const int resident_blocks = sm_count * blocks_per_sm;
    const int blocks = std::min(
        required_blocks, std::min(resident_blocks, kMaxBlocks));

    void* args[] = {
        reinterpret_cast<void*>(&input),
        reinterpret_cast<void*>(&output),
        reinterpret_cast<void*>(&N)
    };
    cudaCheck(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(softmax_kernel<kThreadsPerBlock>),
        dim3(blocks), dim3(kThreadsPerBlock), args, 0, nullptr));
}

std::vector<float> softmax_cpu(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    if (input.empty()) {
        return output;
    }

    const float max_value =
        *std::max_element(input.begin(), input.end());
    double sum = 0.0;
    for (float value : input) {
        sum += std::exp(static_cast<double>(value - max_value));
    }
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = static_cast<float>(
            std::exp(static_cast<double>(input[i] - max_value)) / sum);
    }
    return output;
}

bool check_result(const std::vector<float>& expected,
                  const std::vector<float>& actual) {
    constexpr float atol = 1e-6f;
    constexpr float rtol = 1e-4f;
    int error_count = 0;

    for (size_t i = 0; i < expected.size(); ++i) {
        const float error = std::fabs(expected[i] - actual[i]);
        const float tolerance = atol + rtol * std::fabs(expected[i]);
        if (!std::isfinite(actual[i]) || error > tolerance) {
            if (error_count < 10) {
                std::cerr << "Mismatch at " << i
                          << ": CPU=" << expected[i]
                          << ", GPU=" << actual[i]
                          << ", error=" << error << std::endl;
            }
            ++error_count;
        }
    }
    return error_count == 0;
}

std::vector<int> generate_sizes() {
    return {1, 31, 32, 511, 512, 513, 4096, 1 << 20};
}

int main() {
    cudaCheck(cudaSetDevice(0));

    std::mt19937 generator(0);
    std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);
    bool all_matched = true;

    for (int N : generate_sizes()) {
        std::vector<float> input(N);
        for (float& value : input) {
            value = distribution(generator);
        }

        auto [cpu_output, cpu_time] = benchmark_cpu(softmax_cpu, input);
        std::vector<float> gpu_output(N);

        float* d_input = nullptr;
        float* d_output = nullptr;
        const size_t bytes = static_cast<size_t>(N) * sizeof(float);
        cudaCheck(cudaMalloc(&d_input, bytes));
        cudaCheck(cudaMalloc(&d_output, bytes));
        cudaCheck(cudaMemcpy(
            d_input, input.data(), bytes, cudaMemcpyHostToDevice));

        // Warm up before timing.
        softmax_gpu(d_input, d_output, N);
        cudaCheck(cudaDeviceSynchronize());

        const float gpu_time =
            benchmark_kernel(10, softmax_gpu, d_input, d_output, N);
        cudaCheck(cudaMemcpy(
            gpu_output.data(), d_output, bytes, cudaMemcpyDeviceToHost));

        const bool matched = check_result(cpu_output, gpu_output);
        all_matched = all_matched && matched;
        std::cout << "N=" << N
                  << ", CPU=" << cpu_time.count() << " ms"
                  << ", GPU=" << gpu_time << " ms"
                  << ", matched=" << (matched ? "yes" : "no")
                  << std::endl;

        cudaCheck(cudaFree(d_input));
        cudaCheck(cudaFree(d_output));
    }

    return all_matched ? EXIT_SUCCESS : EXIT_FAILURE;
}
