#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <random>
#include <cmath>
#include <vector>

// Part D entry point, provided by src/flash_attention.cu.
extern "C" void flash_attention_gpu(const float* d_Q, const float* d_K, const float* d_V,
                                    float* d_O, float* d_l, float* d_m,
                                    int B, int H, int N, int D);

static const int BATCH = 8;
static const int HEADS = 16;
static const int HEAD_DIM = 64;
static const int WARMUP_ITERS = 3;
static const int TIMED_ITERS = 10;

// Straightforward O = softmax(Q K^T / sqrt(D)) V, materializing the scores per row.
// Used only to check the kernel; far too slow to run at large N.
void cpu_attention(const float* Q, const float* K, const float* V, float* O,
                   int B, int H, int N, int D) {
    std::vector<float> scores(N);

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            size_t base = ((size_t)b * H + h) * N * D;

            for (int i = 0; i < N; i++) {
                float maxScore = -INFINITY;
                for (int j = 0; j < N; j++) {
                    float dot = 0.0f;
                    for (int x = 0; x < D; x++) {
                        dot += Q[base + (size_t)i * D + x] * K[base + (size_t)j * D + x];
                    }
                    dot /= std::sqrt((float)D);
                    scores[j] = dot;
                    maxScore = std::max(maxScore, dot);
                }

                float sum = 0.0f;
                for (int j = 0; j < N; j++) {
                    scores[j] = std::exp(scores[j] - maxScore);
                    sum += scores[j];
                }

                for (int x = 0; x < D; x++) {
                    float acc = 0.0f;
                    for (int j = 0; j < N; j++) {
                        acc += scores[j] * V[base + (size_t)j * D + x];
                    }
                    O[base + (size_t)i * D + x] = acc / sum;
                }
            }
        }
    }
}

void fill_random(float* A, size_t size, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < size; i++) A[i] = dist(rng);
}

void run_test(int N, bool verify) {
    const int B = BATCH, H = HEADS, D = HEAD_DIM;

    size_t qkvElems = (size_t)B * H * N * D;
    size_t statElems = (size_t)B * H * N;

    float* h_Q = new float[qkvElems];
    float* h_K = new float[qkvElems];
    float* h_V = new float[qkvElems];
    float* h_O = new float[qkvElems];

    fill_random(h_Q, qkvElems, 42);
    fill_random(h_K, qkvElems, 43);
    fill_random(h_V, qkvElems, 44);

    float *d_Q, *d_K, *d_V, *d_O, *d_l, *d_m;
    cudaMalloc(&d_Q, qkvElems * sizeof(float));
    cudaMalloc(&d_K, qkvElems * sizeof(float));
    cudaMalloc(&d_V, qkvElems * sizeof(float));
    cudaMalloc(&d_O, qkvElems * sizeof(float));
    cudaMalloc(&d_l, statElems * sizeof(float));
    cudaMalloc(&d_m, statElems * sizeof(float));

    cudaMemcpy(d_Q, h_Q, qkvElems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qkvElems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, qkvElems * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < WARMUP_ITERS; i++) {
        flash_attention_gpu(d_Q, d_K, d_V, d_O, d_l, d_m, B, H, N, D);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < TIMED_ITERS; i++) {
        flash_attention_gpu(d_Q, d_K, d_V, d_O, d_l, d_m, B, H, N, D);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= TIMED_ITERS;

    cudaMemcpy(h_O, d_O, qkvElems * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << std::setw(7) << N
              << std::setw(14) << std::fixed << std::setprecision(4) << ms << " ms";

    if (verify) {
        float* h_ref = new float[qkvElems];
        cpu_attention(h_Q, h_K, h_V, h_ref, B, H, N, D);

        float maxDiff = 0.0f;
        for (size_t i = 0; i < qkvElems; i++) {
            maxDiff = std::max(maxDiff, std::fabs(h_ref[i] - h_O[i]));
        }
        // The handout's consistency bar across implementations.
        std::cout << "   max abs diff: " << std::scientific << std::setprecision(2) << maxDiff
                  << (maxDiff < 1e-3f ? "  PASSED" : "  FAILED") << std::fixed;
        delete[] h_ref;
    }

    std::cout << "\n";

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "   CUDA error: " << cudaGetErrorString(err) << "\n";
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_l); cudaFree(d_m);
    delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_O;
}

int main() {
    std::cout << "FlashAttention forward pass (CUDA)\n";
    std::cout << "B = " << BATCH << ", heads = " << HEADS << ", D = " << HEAD_DIM
              << ", averaged over " << TIMED_ITERS << " iterations\n\n";
    std::cout << std::setw(7) << "N" << std::setw(17) << "time" << "\n";
    std::cout << "---------------------------------------------------------\n";

    // CPU cross-check only at the small sizes; it is O(N^2 D) per head.
    for (int N = 32; N <= 8192; N *= 2) {
        run_test(N, N <= 128);
    }

    return 0;
}
