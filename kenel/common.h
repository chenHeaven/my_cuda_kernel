//通用运算函数

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <float.h>
#include <cmath>
#include <chrono>
#include <utility>
#include <functional>
// using fp = float;
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&pointer)[0])
#define ceil_div(a, b) ((a + b - 1) / b)
__device__ float& vec_at(float& val, int index) {
    return reinterpret_cast<float*>(&val)[index];
}

__device__ float warp_reduceMax(float val, int warpSize = 32) {
    for (int offset = warpSize / 2; offset > 0; offset /=2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ float warp_reduceSum(float sum, int warpSize = 32) {
    for (int offset = warpSize / 2; offset > 0; offset /=2) {
        sum +=  __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    return sum;
}

bool compare_result(float* cpu, float* gpu, int n, float tol = 1e-6f) {
    int err_num = 0;
    for (int i = 0; i < n; ++i) {
        if (fabsf(cpu[i] - gpu[i]) > tol) {
            err_num++;
        }
        if (err_num > 10) {
            return false;
        }
    }
    return true;
}

void cudacheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("[CUDA ERROR] at file %s %d\n%s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cudacheck(err, __FILE__, __LINE__))

void cublascheck(cublasStatus_t state, const char* file, int line) {
    if (state != CUBLAS_STATUS_SUCCESS) {
        printf("[CUBLAS ERROR] at file %s %d\n%d\n", file, line, state);
        exit(EXIT_FAILURE);
    }
};
#define cublasCheck(state) (cublascheck(state, __FILE__, __LINE__))

template <typename Func, typename... Args>
auto benchmark_cpu(Func&& func, Args&&... args) {
  auto start = std::chrono::high_resolution_clock::now();
  auto result =
            std::forward<Func>(func)(std::forward<Args>(args)...);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  return std::make_pair(result, duration);
}

template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    void* flush_buffer;
    cudaCheck(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    float elapsed_time = 0.f;
    for (int i = 0; i < repeats; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        // now we can start recording the timing of the kernel
        cudaCheck(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        cudaCheck(cudaEventRecord(stop, nullptr));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        float single_call;
        cudaCheck(cudaEventElapsedTime(&single_call, start, stop));
        elapsed_time += single_call;
    }

    cudaCheck(cudaFree(flush_buffer));

    return elapsed_time / repeats;
}

