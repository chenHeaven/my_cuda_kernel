//Implement a program that performs R rounds of parallel hashing on an array of 32-bit integers using 
//the provided hash function. The hash should be applied R times iteratively (the output of one round 
//becomes the input to the next).

#include <cuda_runtime.h>

__device__ unsigned int fnv1a_hash(unsigned int input) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;

    unsigned int hash = OFFSET_BASIS;

    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFFu;
        hash = (hash ^ byte) * FNV_PRIME;
    }
    return hash;
}
__global__ void fnv1a_hash_kernel(const int* __restrict__ input, int* __restrict__ output, int N, int R) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int4* input4 = reinterpret_cast<const int4*>(input);
    uint4* output4 = reinterpret_cast<uint4*>(output);
    if (idx < (N >> 2)) {
        int4 v = __ldg(&input4[idx]);
        uint4 hash4;
        hash4.x = fnv1a_hash(v.x);
        hash4.y = fnv1a_hash(v.y);
        hash4.z = fnv1a_hash(v.z);
        hash4.w = fnv1a_hash(v.w);
        for (int r = 1; r < R; ++r) {
            hash4.x = fnv1a_hash(hash4.x);
            hash4.y = fnv1a_hash(hash4.y);
            hash4.z = fnv1a_hash(hash4.z);
            hash4.w = fnv1a_hash(hash4.w);
        }
        output4[idx] = hash4;
    }
    if (blockIdx.x == 0 && threadIdx.x < (N & 3)) {
        int i = (N & ~3) + threadIdx.x;
        unsigned int hash = fnv1a_hash(__ldg(&input[i]));
        for (int r = 1; r < R; ++r) {
            hash = fnv1a_hash(hash);
        }
        output[i] = hash;
    }
}