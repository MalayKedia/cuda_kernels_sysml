#include "kernels.cuh"

// Batched forward pass: the whole batch of B inputs goes through each kernel in one
// launch, in the default stream. Y = ReLU(X * W1), Z = Y * W2.

#define BLOCK_SIZE 32
#define RELU_THREADS 256

extern "C" void ffnn_forward_gpu(const float* d_X, const float* d_W1, const float* d_W2,
                                 float* d_Y1, float* d_Y2, float* d_Z,
                                 int B, int N) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (B + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Layer 1: Y1 = X * W1
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_W1, d_Y1, B, N, N);

    // Y2 = ReLU(Y1)
    int relu_size = B * N;
    relu_kernel<<<(relu_size + RELU_THREADS - 1) / RELU_THREADS, RELU_THREADS>>>(
        d_Y1, d_Y2, relu_size);

    // Layer 2: Z = Y2 * W2
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Y2, d_W2, d_Z, B, N, N);
}
