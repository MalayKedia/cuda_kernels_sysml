#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <random>
#include <cmath>
#include <vector>

// Forward pass of the two-layer MLP: Y = ReLU(X * W1), Z = Y * W2.
// Provided by whichever src/ffnn_*.cu file this binary was linked against.
extern "C" void ffnn_forward_gpu(const float* d_X, const float* d_W1, const float* d_W2,
                                 float* d_Y1, float* d_Y2, float* d_Z,
                                 int B, int N);

static const int BATCH = 32;
static const int WARMUP_ITERS = 1;
static const int TIMED_ITERS = 10;

void cpu_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float sum = 0;
            for (int t = 0; t < N; t++) {
                sum += A[i * N + t] * B[t * K + j];
            }
            C[i * K + j] = sum;
        }
    }
}

void cpu_ffnn_forward(const float* X, const float* W1, const float* W2, float* Z,
                      int B, int N) {
    std::vector<float> Y((size_t)B * N);
    cpu_matmul(X, W1, Y.data(), B, N, N);
    for (size_t i = 0; i < Y.size(); i++) Y[i] = std::max(0.0f, Y[i]);
    cpu_matmul(Y.data(), W2, Z, B, N, N);
}

void fill_random(float* A, size_t size, unsigned seed) {
    std::mt19937 rng(seed);
    // Centred on zero so ReLU actually gates roughly half the activations.
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < size; i++) A[i] = dist(rng);
}

// The weights dominate memory: two N x N float matrices. Skip any N that will not fit
// alongside the activations, so the sweep degrades gracefully on smaller GPUs.
bool fits_in_memory(int B, int N) {
    size_t weights = 2ull * N * N * sizeof(float);
    size_t acts = 4ull * B * N * sizeof(float);
    size_t needed = weights + acts;

    size_t freeBytes = 0, totalBytes = 0;
    cudaMemGetInfo(&freeBytes, &totalBytes);

    return needed < (size_t)(freeBytes * 0.9);
}

void run_test(int N, bool verify) {
    const int B = BATCH;

    if (!fits_in_memory(B, N)) {
        std::cout << std::setw(8) << N << "   (skipped: does not fit in GPU memory)\n";
        return;
    }

    size_t actElems = (size_t)B * N;
    size_t wElems = (size_t)N * N;

    float* h_X = new float[actElems];
    float* h_W1 = new float[wElems];
    float* h_W2 = new float[wElems];
    float* h_Z = new float[actElems];

    fill_random(h_X, actElems, 42);
    fill_random(h_W1, wElems, 43);
    fill_random(h_W2, wElems, 44);

    float *d_X, *d_W1, *d_W2, *d_Y1, *d_Y2, *d_Z;
    cudaMalloc(&d_X, actElems * sizeof(float));
    cudaMalloc(&d_W1, wElems * sizeof(float));
    cudaMalloc(&d_W2, wElems * sizeof(float));
    cudaMalloc(&d_Y1, actElems * sizeof(float));
    cudaMalloc(&d_Y2, actElems * sizeof(float));
    cudaMalloc(&d_Z, actElems * sizeof(float));

    cudaMemcpy(d_X, h_X, actElems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, h_W1, wElems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, wElems * sizeof(float), cudaMemcpyHostToDevice);

    // Warm up so the timed loop does not pay for context / module load.
    for (int i = 0; i < WARMUP_ITERS; i++) {
        ffnn_forward_gpu(d_X, d_W1, d_W2, d_Y1, d_Y2, d_Z, B, N);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < TIMED_ITERS; i++) {
        ffnn_forward_gpu(d_X, d_W1, d_W2, d_Y1, d_Y2, d_Z, B, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= TIMED_ITERS;

    cudaMemcpy(h_Z, d_Z, actElems * sizeof(float), cudaMemcpyDeviceToHost);

    // Two N x N GEMMs over a batch of B rows.
    double gflops = (4.0 * B * N * N) / (ms * 1e6);

    std::cout << std::setw(8) << N
              << std::setw(14) << std::fixed << std::setprecision(4) << ms << " ms"
              << std::setw(12) << std::setprecision(2) << gflops << " GFLOP/s";

    if (verify) {
        float* h_ref = new float[actElems];
        cpu_ffnn_forward(h_X, h_W1, h_W2, h_ref, B, N);

        bool ok = true;
        double tol = 1e-3 * N; // accumulated float error grows with the reduction length
        for (size_t i = 0; i < actElems; i++) {
            if (std::fabs(h_ref[i] - h_Z[i]) > tol) {
                ok = false;
                break;
            }
        }
        std::cout << "   verify: " << (ok ? "PASSED" : "FAILED");
        delete[] h_ref;
    }

    std::cout << "\n";

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "   CUDA error: " << cudaGetErrorString(err) << "\n";
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_X);
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_Y1);
    cudaFree(d_Y2);
    cudaFree(d_Z);

    delete[] h_X;
    delete[] h_W1;
    delete[] h_W2;
    delete[] h_Z;
}

int main() {
    std::cout << "Two-layer MLP forward pass, batch size B = " << BATCH << "\n";
    std::cout << "averaged over " << TIMED_ITERS << " iterations after "
              << WARMUP_ITERS << " warmup\n\n";
    std::cout << std::setw(8) << "N" << std::setw(17) << "time"
              << std::setw(20) << "throughput" << "\n";
    std::cout << "-------------------------------------------------------------\n";

    // N from 32 to 32K, doubling. CPU verification only where it is cheap.
    for (int N = 32; N <= 32768; N *= 2) {
        run_test(N, N <= 512);
    }

    return 0;
}
