#include "kernels.cuh"

// Stream-parallel forward pass. The batch is split into NUM_STREAMS chunks of B/NUM_STREAMS
// rows; each chunk runs its own matmul -> ReLU -> matmul chain in its own CUDA stream, so
// the three dependent kernels of one chunk overlap with those of the others. Chunks are
// independent (the weights are read-only and each chunk owns its slice of Y1/Y2/Z), so no
// cross-stream synchronization is needed until the whole batch is done.

#define BLOCK_SIZE 32
#define RELU_THREADS 256
#define NUM_STREAMS 4

extern "C" void ffnn_forward_gpu(const float* d_X, const float* d_W1, const float* d_W2,
                                 float* d_Y1, float* d_Y2, float* d_Z,
                                 int B, int N) {
    static cudaStream_t streams[NUM_STREAMS];
    static bool initialized = false;
    if (!initialized) {
        for (int s = 0; s < NUM_STREAMS; s++) cudaStreamCreate(&streams[s]);
        initialized = true;
    }

    // Split the batch as evenly as possible when B is not a multiple of NUM_STREAMS.
    int base = B / NUM_STREAMS;
    int rem = B % NUM_STREAMS;

    int rowOffset = 0;
    for (int s = 0; s < NUM_STREAMS; s++) {
        int rows = base + (s < rem ? 1 : 0);
        if (rows == 0) continue;

        // Each chunk owns rows [rowOffset, rowOffset + rows) of X, Y1, Y2 and Z.
        const float* X_chunk = d_X + (size_t)rowOffset * N;
        float* Y1_chunk = d_Y1 + (size_t)rowOffset * N;
        float* Y2_chunk = d_Y2 + (size_t)rowOffset * N;
        float* Z_chunk = d_Z + (size_t)rowOffset * N;

        dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                           (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

        matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock, 0, streams[s]>>>(
            X_chunk, d_W1, Y1_chunk, rows, N, N);

        int relu_size = rows * N;
        relu_kernel<<<(relu_size + RELU_THREADS - 1) / RELU_THREADS, RELU_THREADS, 0, streams[s]>>>(
            Y1_chunk, Y2_chunk, relu_size);

        matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock, 0, streams[s]>>>(
            Y2_chunk, d_W2, Z_chunk, rows, N, N);

        rowOffset += rows;
    }

    for (int s = 0; s < NUM_STREAMS; s++) cudaStreamSynchronize(streams[s]);
}
