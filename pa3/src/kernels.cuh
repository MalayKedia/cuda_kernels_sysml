#pragma once

#include <cuda_runtime.h>

// C = A * B, with A of shape M x N and B of shape N x K, all row-major.
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C,
                                             int M, int N, int K);

__global__ void relu_kernel(const float* A, float* C, int size);

// Forward pass of the two-layer MLP: Y = ReLU(X * W1), Z = Y * W2.
// X is B x N, W1 and W2 are N x N, Y1/Y2/Z are B x N scratch/output buffers.
extern "C" void ffnn_forward_gpu(const float* d_X, const float* d_W1, const float* d_W2,
                                 float* d_Y1, float* d_Y2, float* d_Z,
                                 int B, int N);
