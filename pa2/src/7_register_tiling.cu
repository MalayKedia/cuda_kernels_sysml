#include <cuda_runtime.h>

// Part C idea: minimize shared memory usage by keeping one operand in registers.
// Only the BK x BN tile of B goes through shared memory; each thread holds its
// TM values of A directly in registers, read straight from global memory. This
// halves shared-memory traffic and footprint relative to 6_tiling_2d, at the
// cost of A being re-read from L2/global by every block along a row.

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float Bs[BK][BN];

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

    for (int t = 0; t < numTiles; t++) {
        int kBase = t * BK;

        for (int idx = tid; idx < BK * BN; idx += numThreads) {
            int r = idx / BN;
            int c = idx % BN;
            int gRow = kBase + r;
            int gCol = blockCol + c;
            Bs[r][c] = (gRow < N && gCol < K) ? B[gRow * K + gCol] : 0.0f;
        }

        __syncthreads();

        // A stays in registers: TM rows x BK reduction steps for this thread.
        float regA[TM][BK];
        #pragma unroll
        for (int i = 0; i < TM; i++) {
            int gRow = blockRow + ty * TM + i;
            #pragma unroll
            for (int kk = 0; kk < BK; kk++) {
                int gCol = kBase + kk;
                regA[i][kk] = (gRow < M && gCol < N) ? A[gRow * N + gCol] : 0.0f;
            }
        }

        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float regB[TN];
            #pragma unroll
            for (int j = 0; j < TN; j++) regB[j] = Bs[kk][tx * TN + j];

            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += regA[i][kk] * regB[j];
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
