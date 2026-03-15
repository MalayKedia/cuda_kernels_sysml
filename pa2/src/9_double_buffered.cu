#include <cuda_runtime.h>

// Double-buffered 2D tiling. Two shared-memory tile buffers are alternated: while
// the block computes on buffer t%2, the global loads for tile t+1 are already in
// flight into buffer (t+1)%2. Because the two buffers never alias within an
// iteration, only one __syncthreads per tile is needed instead of two, and the
// memory latency of the next tile overlaps with the FMAs of the current one.

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[2][BK][BM]; // transposed: As[buf][k][m]
    __shared__ float Bs[2][BK][BN];

    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * (BN / TN) + tx;
    const int numThreads = (BM / TM) * (BN / TN);

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    int numTiles = (N + BK - 1) / BK;

    // Cooperative load of tile `t` into shared buffer `buf`.
    #define LOAD_TILE(t, buf)                                                        \
        {                                                                            \
            int kBase = (t) * BK;                                                    \
            for (int idx = tid; idx < BM * BK; idx += numThreads) {                  \
                int r = idx / BK;                                                    \
                int c = idx % BK;                                                    \
                int gRow = blockRow + r;                                             \
                int gCol = kBase + c;                                                \
                As[buf][c][r] = (gRow < M && gCol < N) ? A[gRow * N + gCol] : 0.0f;  \
            }                                                                        \
            for (int idx = tid; idx < BK * BN; idx += numThreads) {                  \
                int r = idx / BN;                                                    \
                int c = idx % BN;                                                    \
                int gRow = kBase + r;                                                \
                int gCol = blockCol + c;                                             \
                Bs[buf][r][c] = (gRow < N && gCol < K) ? B[gRow * K + gCol] : 0.0f;  \
            }                                                                        \
        }

    LOAD_TILE(0, 0);
    __syncthreads();

    for (int t = 0; t < numTiles; t++) {
        int cur = t & 1;

        if (t + 1 < numTiles) {
            LOAD_TILE(t + 1, cur ^ 1);
        }

        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float regA[TM];
            float regB[TN];

            #pragma unroll
            for (int i = 0; i < TM; i++) regA[i] = As[cur][kk][ty * TM + i];

            #pragma unroll
            for (int j = 0; j < TN; j++) regB[j] = Bs[cur][kk][tx * TN + j];

            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += regA[i] * regB[j];
        }

        __syncthreads();
    }

    #undef LOAD_TILE

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int gRow = blockRow + ty * TM + i;
        if (gRow >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int gCol = blockCol + tx * TN + j;
            if (gCol < K) C[gRow * K + gCol] = acc[i][j];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void matmul_gpu(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(BN / TN, BM / TM);
    dim3 blocksPerGrid((K + BN - 1) / BN, (M + BM - 1) / BM);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
