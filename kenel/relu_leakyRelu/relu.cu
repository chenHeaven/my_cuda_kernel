__global__ void relu_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= N) return;

    output[idx] = max(0.0f, input[idx]);
}

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= N) return;
    float alpha = 0.01f;
    float val = input[idx];
    output[idx] = (val > 0) ? val : alpha * val;
}