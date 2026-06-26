#include <cuda_runtime.h>

__constant__ float kernel_const[31 * 31];
#define TILE 32
#define MAXK 32
__global__ void convolution(
    const float* __restrict__ input,
    float* __restrict__ output,
    int ir, int ic, int o_r, int oc, int kr, int kc) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int ix = blockIdx.x * TILE + tx;
    int iy = blockIdx.y * TILE + ty;

    __shared__ float smem[TILE + MAXK][TILE + MAXK];

    for (int dy = 0; (ty + dy) < (TILE + MAXK) && (iy + dy) < ir; dy += TILE) {
        for (int dx = 0; (tx + dx) < (TILE + MAXK) && (ix + dx) < ic; dx += TILE) {
            smem[ty + dy][tx + dx] = input[(iy + dy) * ic + dx + ix];
        }
    }
    __syncthreads();
    float sum = 0.0f;
    if (ix < oc && iy < o_r) {
        for (int ky = 0; ky < kr; ++ky) {
            for (int kx = 0; kx < kc; ++kx) {
                sum += smem[ty + ky][tx + kx] * kernel_const[ky * kc + kx];
            }
        }
        output[iy * oc + ix] = sum;
    }
}
#define ceil(a, b) ((a + b - 1) / b)
// input, kernel, output are device pointers
void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
    cudaMemcpyToSymbol(kernel_const, kernel, kernel_rows * kernel_cols * sizeof(float));

    int out_row = input_rows - kernel_rows + 1;
    int out_col = input_cols - kernel_cols + 1;

    dim3 block_size(TILE, TILE);
    dim3 grid_size(ceil(out_col, TILE), ceil(out_row, TILE));
    convolution<<<grid_size, block_size>>>(input, output, input_rows, input_cols, out_row, out_col, kernel_rows, kernel_cols);
}
