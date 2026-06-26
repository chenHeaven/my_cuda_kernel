#include <cuda_runtime.h>


__global__ void pool(const float* input, float* output, int N, int C, int H, int W, int H_out, int W_out,
                      int K_SIZE, int stride, int padding) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int channel = threadIdx.y + blockDim.y * blockIdx.y;
    int batch = threadIdx.z + blockDim.z * blockIdx.z;

    if (batch < N && channel < C && idx < H_out * W_out) {
        int h_nake = idx / W_out;
        int w_nake = idx % W_out;

        int h_start = h_nake * stride - padding;
        int w_start = w_nake * stride - padding;

        float max_val = -__FLT_MAX__;
        int start = batch * C * H * W + channel * H * W;
#pragma unroll
        for (int hi = 0; hi < K_SIZE; ++hi) {
#pragma unroll
            for (int wi = 0; wi < K_SIZE; ++wi) {
                int h = h_start + hi;
                int w = w_start + wi;

                float curr = 0.0f;
                if (h >=0 && h < H && w >= 0 && w < W) {
                    curr = input[start + h * W + w];
                    if (curr > max_val) max_val = curr;
                }
            }
        }
        output[batch * C * H_out * W_out + channel * H_out * W_out + idx] = max_val;
    }

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
#define ceil(a, b) ((a + b - 1) / b)
extern "C" void solve(const float* input, float* output, int N, int C, int H, int W,
                      int kernel_size, int stride, int padding) {
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    dim3 block_size(256, 1, 1);
    dim3 grid_size(ceil(H_out * W_out, block_size.x), ceil(C, block_size.y), ceil(N, block_size.z));

    pool<<<grid_size, block_size>>>(input, output, N, C, H, W, H_out, W_out, kernel_size, stride, padding);
}
