#include <cuda_runtime.h>

// 2D shared-memory + register tiling: each thread block loads a BM x BK tile of A
// and a BK x BN tile of B into shared memory; each thread then accumulates a
// TM x TN sub-tile of C in registers, reusing every value it reads from shared
// memory TN (or TM) times before moving to the next reduction step.

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    int tx = threadIdx.x; // 0 .. BN/TN - 1
    int ty = threadIdx.y; // 0 .. BM/TM - 1
    int tid = ty * (BN / TN) + tx;
    const int numThreads = (BM / TM) * (BN / TN);

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    int numTiles = (N + BK - 1) / BK;

    for (int t = 0; t < numTiles; t++) {
        int kBase = t * BK;

        // Cooperatively load the BM x BK tile of A.
        for (int idx = tid; idx < BM * BK; idx += numThreads) {
            int r = idx / BK;
            int c = idx % BK;
            int gRow = blockRow + r;
            int gCol = kBase + c;
            As[r][c] = (gRow < M && gCol < N) ? A[gRow * N + gCol] : 0.0f;
        }

        // Cooperatively load the BK x BN tile of B.
        for (int idx = tid; idx < BK * BN; idx += numThreads) {
            int r = idx / BN;
            int c = idx % BN;
            int gRow = kBase + r;
            int gCol = blockCol + c;
            Bs[r][c] = (gRow < N && gCol < K) ? B[gRow * K + gCol] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float regA[TM];
            float regB[TN];

            #pragma unroll
            for (int i = 0; i < TM; i++) regA[i] = As[ty * TM + i][kk];

            #pragma unroll
            for (int j = 0; j < TN; j++) regB[j] = Bs[kk][tx * TN + j];

            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += regA[i] * regB[j];
        }

        __syncthreads();
    }

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
