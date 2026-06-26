#include <cuda_runtime.h>
#include <cmath>
#include <cooperative_groups.h>
#include <cstdlib>
#include <iostream>
#include <algorithm>
namespace cg = cooperative_groups;
// #define eps 1e-5
#define TILE 32
#define threadPerBlock 1024

constexpr float ceil(float a, float b) {
    return (a + b - 1) / b;
}

__inline__ __device__ float block_reduce(float val) {
  const int tid = threadIdx.x;
  const int warpSize = 32;
  int lane = tid % warpSize;
  int warp_id = tid / warpSize;

  // Warp-level reduction
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);

  // Write warp result to shared memory
  __shared__ float warpSums[32];  // Max 32 warps per block
  if (lane == 0) {
    warpSums[warp_id] = val;
  }
  __syncthreads();

  // Final reduction: only first warp participates
  if (warp_id == 0) {
    val = (tid < (blockDim.x + warpSize - 1) / warpSize) ? warpSums[tid] : 0.0f;
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
      val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  } else {
    val = 0.0f;
  }
  return val;
}
__device__ float globalMem = 0.0f;
__global__ void RMSNorm(const float* input, float gamma, float beta, float* output, int N, float eps) {
    cg::grid_group grid = cg::this_grid();

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    const float4* inputV4 = reinterpret_cast<const float4*>(input);
    
    float4* outputV4 = reinterpret_cast<float4*>(output);
    int numVec = N / 4;
    float sum = 0.0f;
    for (int i = tid; i < numVec; i += blockDim.x * gridDim.x) {
        float4 tempV4 = inputV4[i];
        sum += tempV4.x * tempV4.x;
        sum += tempV4.y * tempV4.y;
        sum += tempV4.z * tempV4.z;
        sum += tempV4.w * tempV4.w;
    }
    for (int i = (numVec << 2) + tid; i < N; i += blockDim.x * gridDim.x) {
        sum += input[i] * input[i];
    }
    sum = block_reduce(sum);
    // __shared__ float smem;
    if ((threadIdx.x == 0)) atomicAdd(&globalMem, sum);
    grid.sync();
    // __syncthreads();
    const float scale = rsqrtf(globalMem / static_cast<float>(N) + eps);
    
    for (int i = tid; i < numVec; i += blockDim.x * gridDim.x) {
        float4 tempV4 = inputV4[i];
        outputV4[i] = make_float4((scale * tempV4.x * gamma + beta), (scale * tempV4.y * gamma + beta), (scale * tempV4.z * gamma + beta), (scale * tempV4.w * gamma + beta));
    }
    for (int i = (numVec << 2) + tid; i < N; i += blockDim.x * gridDim.x) {
        output[i] = scale * input[i] * gamma + beta;
    }

}


// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N,
                      float eps) {

    float zero = 0.0f;
    cudaMemcpyToSymbol(globalMem, &zero, sizeof(float));
    int block_size = threadPerBlock;
    
    int device = 0;
    cudaGetDevice(&device);
    int cooperative_launch = 0;
    cudaDeviceGetAttribute(&cooperative_launch, cudaDevAttrCooperativeLaunch, device);
    if (!cooperative_launch) {
        std::cerr << "Current GPU does not support cooperative kernel launch." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, RMSNorm, block_size, 0);
    int required_blocks = (N + block_size - 1) / block_size;
    int resident_blocks = sm_count * blocks_per_sm;
    int grid_size = std::min(required_blocks, resident_blocks);
    void* args[] = {
        reinterpret_cast<void*>(&input),
        reinterpret_cast<void*>(&gamma),
        reinterpret_cast<void*>(&beta),
        reinterpret_cast<void*>(&output),
        reinterpret_cast<void*>(&N),
        reinterpret_cast<void*>(&eps)
    };
    cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(RMSNorm), dim3(grid_size), dim3(block_size),
        args, 0, nullptr);
}
