#include "kernels.cuh"
#include <cublas_v2.h>

// Same forward pass with the two GEMMs handed to cuBLAS, as a ceiling to measure the
// hand-written kernels against. cuBLAS is column-major and the buffers here are
// row-major, so a row-major (B x N) * (N x N) is issued as its column-major transpose
// by passing the weight matrix first with the dimensions swapped.

#define RELU_THREADS 256

extern "C" void ffnn_forward_gpu(const float* d_X, const float* d_W1, const float* d_W2,
                                 float* d_Y1, float* d_Y2, float* d_Z,
                                 int B, int N) {
    static cublasHandle_t handle = nullptr;
    if (!handle) cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Y1 = X * W1
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, B, N, &alpha, d_W1, N, d_X, N, &beta, d_Y1, N);

    int relu_size = B * N;
    relu_kernel<<<(relu_size + RELU_THREADS - 1) / RELU_THREADS, RELU_THREADS>>>(
        d_Y1, d_Y2, relu_size);

    // Z = Y2 * W2
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, B, N, &alpha, d_W2, N, d_Y2, N, &beta, d_Z, N);
}
