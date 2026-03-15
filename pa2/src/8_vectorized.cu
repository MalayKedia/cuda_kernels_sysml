#include <cuda_runtime.h>

// Vectorized (float4) global loads on top of 2D tiling, with the A tile stored
// transposed in shared memory so the inner loop reads As[kk][...] contiguously
// instead of striding down a column. Larger tiles (128x128 with an 8x8 register
// block) raise arithmetic intensity further.
//
// float4 loads require the row strides to be multiples of 4 and the tile to be
// fully in bounds, so a scalar fallback path keeps the ragged/small cases correct.

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BK][BM]; // transposed: As[k][m]
    __shared__ float Bs[BK][BN];

    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * (BN / TN) + tx;
    const int numThreads = (BM / TM) * (BN / TN);

    bool vectorizable = (N % 4 == 0) && (K % 4 == 0) &&
                        (blockRow + BM <= M) && (blockCol + BN <= K);

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    int numTiles = (N + BK - 1) / BK;

    for (int t = 0; t < numTiles; t++) {
        int kBase = t * BK;

        if (vectorizable && kBase + BK <= N) {
            // Each thread pulls one float4 of A and one of B per tile.
            for (int idx = tid; idx < (BM * BK) / 4; idx += numThreads) {
                int r = idx / (BK / 4);
                int c = (idx % (BK / 4)) * 4;
                float4 v = *reinterpret_cast<const float4*>(&A[(blockRow + r) * N + kBase + c]);
                As[c + 0][r] = v.x;
                As[c + 1][r] = v.y;
                As[c + 2][r] = v.z;
                As[c + 3][r] = v.w;
            }

            for (int idx = tid; idx < (BK * BN) / 4; idx += numThreads) {
                int r = idx / (BN / 4);
                int c = (idx % (BN / 4)) * 4;
                *reinterpret_cast<float4*>(&Bs[r][c]) =
                    *reinterpret_cast<const float4*>(&B[(kBase + r) * K + blockCol + c]);
            }
        } else {
            for (int idx = tid; idx < BM * BK; idx += numThreads) {
                int r = idx / BK;
                int c = idx % BK;
                int gRow = blockRow + r;
                int gCol = kBase + c;
                As[c][r] = (gRow < M && gCol < N) ? A[gRow * N + gCol] : 0.0f;
            }

            for (int idx = tid; idx < BK * BN; idx += numThreads) {
                int r = idx / BN;
                int c = idx % BN;
                int gRow = kBase + r;
                int gCol = blockCol + c;
                Bs[r][c] = (gRow < N && gCol < K) ? B[gRow * K + gCol] : 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float regA[TM];
            float regB[TN];

            #pragma unroll
            for (int i = 0; i < TM; i++) regA[i] = As[kk][ty * TM + i];

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
