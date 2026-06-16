//矩阵乘法


#include <iostream>
#include "../common.h"
#include <fstream>
#include <cublas_v2.h>


#define OFFSET(row, col, n) ((row) * (n) + (col))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void matmul_v1(int M, int N, int K, const float* A, const float* B, float* C,float alpha, float beta) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    constexpr int block_row_thread = BN / TN;
    constexpr int block_col_thread = BM / TM;
    constexpr int block_num_tread = block_row_thread * block_col_thread;

    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;

    constexpr int ldg_a_num = BM * BK / block_num_tread / 4;      //每个线程处理多少个float4
    constexpr int ldg_b_num = BN * BK / block_num_tread / 4;

    int a_tile_row = threadIdx.x / (BK / 4);
    int a_tile_col = threadIdx.x % (BK / 4) * 4;
    constexpr int a_tile_stride = BM / ldg_a_num;                  //在BM维度的每个线程需要的偏移

    __shared__ float a_shared[BM * BK];
    __shared__ float b_shared[BK * BN];

    int b_tile_row = threadIdx.x / (BN / 4);
    int b_tile_col = threadIdx.x % (BN / 4) * 4;
    constexpr int b_tile_stride = BK / ldg_b_num;

    float ldg_a_reg[4 * ldg_a_num] = {0.f};
    float sum[TM][TN] = {0.};
    float a_frag[TM];
    float b_frag[TN];
    A = &A[by * K * BM];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    for (int k = 0; k < K; k += BK) {
        for (int i = 0; i < BM; i += a_tile_stride) {
            int index = i / a_tile_stride * 4;
            FLOAT4(ldg_a_reg[index]) = reinterpret_cast<const float4*>(&A[OFFSET(i + a_tile_row, a_tile_col, K)])[0];
            a_shared[OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[index];
            a_shared[OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[index + 1];
            a_shared[OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[index + 2];
            a_shared[OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[index + 3];
        }
        for (int j = 0; j < BK; j += b_tile_stride) {
            FLOAT4(b_shared[OFFSET(j + b_tile_row, b_tile_col, BN)]) =
                reinterpret_cast<const float4*>(&B[OFFSET(j + b_tile_row, b_tile_col, N)])[0];
        }
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int i = 0; i < BK; ++i) {
            for (int m = 0; m < TM; m += 4) {
                FLOAT4(a_frag[m]) = FLOAT4(a_shared[OFFSET(i, ty + m, BM)]);
            }
            for (int n = 0; n < TN; n += 4) {
                FLOAT4(b_frag[n]) = FLOAT4(b_shared[OFFSET(i, tx + n, BN)]);
            }
            for (int m = 0; m < TM; m++) {
                for (int n = 0; n < TN; n++) {
                    sum[m][n] += a_frag[m] * b_frag[n];
                }
            }
        }
        __syncthreads();
    }
    for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n += 4) {
            float4 cmpt = FLOAT4(C[OFFSET(ty + m, tx + n, N)]);
            cmpt.x = alpha * sum[m][n] + beta * cmpt.x;
            cmpt.y = alpha * sum[m][n + 1] + beta * cmpt.y;
            cmpt.z = alpha * sum[m][n + 2] + beta * cmpt.z;
            cmpt.w = alpha * sum[m][n + 3] + beta * cmpt.w;
            FLOAT4(C[OFFSET(ty + m, tx + n, N)]) = cmpt;
        }
    }
}




#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)
std::vector<int> generateSizes() { return {4096}; }
int main() {
  int device_id = 0;
  cudaCheck(cudaSetDevice(device_id));
  std::vector<int> sizes = generateSizes();
  // 打开CSV文件
  std::ofstream csv_file("sgemm_benchmark_v4.csv");
  csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

  for (int N : sizes) {
    std::cout << "Testing size: " << N << std::endl;

    size_t size = N * N * sizeof(float);
    float* A = (float*)malloc(size);
    float* B = (float*)malloc(size);
    float* C_cublas = (float*)malloc(size);
    float* C_v1 = (float*)malloc(size);

    float *d_A, *d_B, *d_C_v1;
    cudaCheck(cudaMalloc(&d_A, size));
    cudaCheck(cudaMalloc(&d_B, size));
    cudaCheck(cudaMalloc(&d_C_v1, size));

    bool out_of_memory = false;

    try {
      // 初始化矩阵 A 和 B
      for (int i = 0; i < N * N; ++i) {
        A[i] = 1.0f;
        B[i] = 2.0f;
      }
            // 拷贝到设备
      cudaCheck(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
      cudaCheck(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

      cublasHandle_t handle;
      cublasCheck(cublasCreate(&handle));
      float alpha = 1.0f;
      float beta = 0.0f;
      float cublas_time = benchmark_kernel(10, [&]() {
        cublasCheck(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                &alpha, d_B, N, d_A, N, &beta, d_C_v1, N));
      });

      cudaCheck(cudaMemcpy(C_cublas, d_C_v1, size, cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemset(d_C_v1, 0, size));
      dim3 blockDim(256);
      dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));
      float v1_time = benchmark_kernel(10, [&]() {
        matmul_v1<128, 128, 8, 8, 8>
            <<<gridDim, blockDim>>>(N, N, N, d_A, d_B, d_C_v1, alpha, beta);
        cudaCheck(cudaGetLastError());
      });

      // 拷贝手写 kernel 结果
      cudaCheck(cudaMemcpy(C_v1, d_C_v1, size, cudaMemcpyDeviceToHost));
      // 结果比较
      int error_count = 0;
      float TOL = 1e-6f;
      for (int i = 0; i < N * N && error_count < 10; ++i) {
        if (fabsf(C_cublas[i] - C_v1[i]) > TOL) {
          error_count++;
        }
      }
      int repeat_time = 10;
      float cublas_gflops =
          repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f);  // GFlops
      float v1_gflops =
          repeat_time * 2.0f * N * N * N / (v1_time * 1e6f);  // GFlops
      // 写入CSV
      csv_file << N << "," << cublas_gflops << "," << v1_gflops << ","
               << (error_count == 0 ? "1" : "0") << std::endl;
    } catch (...) {
      std::cerr << "Out of memory or error during testing size: " << N
                << std::endl;
      out_of_memory = true;
    }

    if (!out_of_memory) {
      std::cout << "Finished size: " << N << std::endl;
    } else {
      csv_file << N << ",OOM,OOM,0" << std::endl;
    }
  }

  csv_file.close();

  std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark.csv'"
            << std::endl;
  return 0;
}
