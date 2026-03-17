#include "kernels.cuh"

__global__ void relu_kernel(const float* A, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = fmaxf(A[idx], 0.0f);
    }
}