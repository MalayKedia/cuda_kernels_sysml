#include "kernels.cuh"

// Shared-memory tiled matmul. The MLP's activation matrices are only B = 32 rows
// tall, so the aggressive multi-output-per-thread tiling used in PA2 would leave
// most of each thread's register block out of bounds. One output per thread with a
// 32x32 tile keeps every thread doing useful work at this shape.

#define TILE_SIZE 32

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C,
                                             int M, int N, int K) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int globalCol = blockIdx.x * TILE_SIZE + threadIdx.x;
    int globalRow = blockIdx.y * TILE_SIZE + threadIdx.y;

    int localCol = threadIdx.x;
    int localRow = threadIdx.y;

    float sum = 0.0f;

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + localCol;
        int bRow = t * TILE_SIZE + localRow;

        tile_A[localRow][localCol] =
            (globalRow < M && aCol < N) ? A[globalRow * N + aCol] : 0.0f;
        tile_B[localRow][localCol] =
            (bRow < N && globalCol < K) ? B[bRow * K + globalCol] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tile_A[localRow][i] * tile_B[i][localCol];
        }

        __syncthreads();
    }

    if (globalRow < M && globalCol < K) {
        C[globalRow * K + globalCol] = sum;
    }
}
