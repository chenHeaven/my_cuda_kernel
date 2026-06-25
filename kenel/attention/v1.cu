// 普通的softmax attention函数的实现
#include <cuda_runtime.h>
#include <cmath>

// transpose
constexpr int kTransposeTile = 32;
constexpr int kTransposeRows = 8;

__device__ __forceinline__ float warp_reduceMax(float val, int warpSize = 32) {
    for (int offset = warpSize / 2; offset > 0; offset /=2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduceSum(float sum, int warpSize = 32) {
    for (int offset = warpSize / 2; offset > 0; offset /=2) {
        sum +=  __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    return sum;
}

__global__ void transpose(const float* input, float* output, int rows, int cols) {
    int idx = blockIdx.x * kTransposeTile + threadIdx.x;
    int idy = blockIdx.y * kTransposeTile + threadIdx.y;
    __shared__ float shared[kTransposeTile][kTransposeTile + 1];

    for (int i = 0; i < kTransposeTile; i += kTransposeRows) {
        int indexY = idy + i;
        if (idx < cols && indexY < rows) {
            shared[threadIdx.y + i][threadIdx.x] = input[indexY * cols + idx];
        }
    }
    __syncthreads();
    idx = blockIdx.y * kTransposeTile + threadIdx.x;
    idy = blockIdx.x * kTransposeTile + threadIdx.y;
    for (int i = 0; i < kTransposeTile; i += kTransposeRows) {
        int indexY = idy + i;
        if (idx < rows && indexY < cols) {
            output[indexY * rows + idx] = shared[threadIdx.x][threadIdx.y + i];
        }
    }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void matmul_v1(int M, int N, int K, const float* A, const float* B, float* C, float alpha, float beta) {
    constexpr int kThreadCols = BN / TN;
    constexpr int kThreadRows = BM / TM;
    constexpr int kThreadCount = kThreadRows * kThreadCols;
    static_assert(BM % TM == 0 && BN % TN == 0);
    static_assert(kThreadCount == 256);

    // A is stored transposed in shared memory so each warp reads it without
    // shared-memory bank conflicts during the inner-product loop.
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int tid = threadIdx.x;
    const int tx = (tid % kThreadCols) * TN;
    const int ty = (tid / kThreadCols) * TM;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    float a_frag[TM];
    float b_frag[TN];
    float sum[TM][TN] = {0.0f};

    for (int k_base = 0; k_base < K; k_base += BK) {
        for (int index = tid; index < BM * BK; index += kThreadCount) {
            const int row = index / BK;
            const int k = index % BK;
            const int global_row = block_row + row;
            const int global_k = k_base + k;
            As[k * BM + row] =
                global_row < M && global_k < K
                    ? A[global_row * K + global_k]
                    : 0.0f;
        }
        for (int index = tid; index < BK * BN; index += kThreadCount) {
            const int k = index / BN;
            const int col = index % BN;
            const int global_k = k_base + k;
            const int global_col = block_col + col;
            Bs[k * BN + col] =
                global_k < K && global_col < N
                    ? B[global_k * N + global_col]
                    : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                a_frag[m] = As[k * BM + ty + m];
            }
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                b_frag[n] = Bs[k * BN + tx + n];
            }
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    sum[m][n] += a_frag[m] * b_frag[n];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        const int row = block_row + ty + m;
        if (row >= M) {
            continue;
        }
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
            const int col = block_col + tx + n;
            if (col < N) {
                const int index = row * N + col;
                C[index] = beta == 0.0f
                    ? alpha * sum[m][n]
                    : alpha * sum[m][n] + beta * C[index];
            }
        }
    }
}

__device__ __forceinline__ void reduceBlockSum(const float* inp, float* out, int n, int warpSize = 32) {
    int id = threadIdx.x;
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    float sum = 0.0f;
    __shared__ float warpNum[32];
    for (int i = id; i < n; i += blockDim.x) {
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
        *out = sum;
    }
    __syncthreads();
}

__device__ __forceinline__ void reduceBlockMax(const float* inp, float* out, int n, int warpSize = 32) {
    int id = threadIdx.x;
    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    float val = -INFINITY;
    __shared__ float warpNum[32];
    for (int i = id; i < n; i += blockDim.x) {
        val = fmaxf(val, inp[i]);
    }
    val = warp_reduceMax(val);
    if (lane == 0) {
        warpNum[warpId] = val;
    }
    __syncthreads();
    if (warpId == 0) {
        val = (threadIdx.x < blockDim.x / warpSize) ? warpNum[threadIdx.x] : -INFINITY;
        val = warp_reduceMax(val);
    }
    if (threadIdx.x == 0) {
        *out = val;
    }
    __syncthreads();
}

__global__ void softmax_naive(float* input, float* output, float scale, int M, int N) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    if (bid >= M) {
        return;
    }
    __shared__ float b_max, b_sum;
    float* row_input = input + N * bid;

    for (int i = tid; i < N; i += blockDim.x) {
        row_input[i] = row_input[i] * scale;
    }
    __syncthreads();
    reduceBlockMax(row_input, &b_max, N);

    for (int i = tid; i < N; i += blockDim.x) {
        row_input[i] = __expf(row_input[i] - b_max);
    }
    __syncthreads();
    reduceBlockSum(row_input, &b_sum, N);
    if (tid == 0) {
        b_sum = 1.0f / b_sum;
    }
    __syncthreads();
    for (int i = tid; i < N; i += blockDim.x) {
        output[bid * N + i] = b_sum * row_input[i];
    }
}

constexpr int ceil_div(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

// Q, K, V, output are device pointers
void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d) {
    if (M <= 0 || N <= 0 || d <= 0) {
        return;
    }

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    float *KT, *score, *soft;
    cudaMalloc(
        &KT, static_cast<size_t>(d) * static_cast<size_t>(N) * sizeof(float));
    cudaMalloc(
        &score,
        static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float));
    cudaMalloc(
        &soft,
        static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float));
    dim3 transpose_block_size(kTransposeTile, kTransposeRows);
    dim3 transpose_grid_size(
        ceil_div(d, kTransposeTile), ceil_div(N, kTransposeTile));
    transpose<<<transpose_grid_size, transpose_block_size>>>(K, KT, N, d);

    constexpr int kMatmulTile = 128;
    dim3 matmul_block_size(256);
    dim3 matmul_grid_size(
        ceil_div(N, kMatmulTile), ceil_div(M, kMatmulTile));
    matmul_v1<128, 128, 8, 8, 8><<<matmul_grid_size, matmul_block_size>>>(M, N, d, Q, KT, score, alpha, beta);

    constexpr int soft_block_size = 256;
    const int soft_grid_size = M;
    float scale = 1.0f / sqrtf(static_cast<float>(d));
    softmax_naive<<<soft_grid_size, soft_block_size>>>(score, soft, scale, M, N);

    matmul_grid_size =
        dim3(ceil_div(d, kMatmulTile), ceil_div(M, kMatmulTile));
    matmul_v1<128, 128, 8, 8, 8><<<matmul_grid_size, matmul_block_size>>>(M, d, N, soft, V, output, alpha, beta);
    cudaFree(KT);
    cudaFree(score);
    cudaFree(soft);
}
