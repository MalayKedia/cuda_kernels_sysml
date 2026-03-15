#include <cuda_runtime.h>
#include <cublas_v2.h>

// cuBLAS SGEMM driven through the same harness as the hand-written kernels, so its
// number is measured identically (same CUDA events, same inputs) rather than only
// as the side reference main.cc prints.
//
// cuBLAS is column-major while the host buffers are row-major. A row-major C = A*B
// is the same bytes as a column-major C^T = B^T * A^T, so passing B and A in that
// order with the swapped dimensions gives the right result without transposing.

extern "C" void matmul_gpu(const float* A, const float* B, float* C, int M, int N, int K) {
    static cublasHandle_t handle = nullptr;
    if (!handle) cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                K, M, N,
                &alpha,
                B, K,
                A, N,
                &beta,
                C, K);

    cudaDeviceSynchronize();
}
