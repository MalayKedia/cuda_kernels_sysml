#include <cuda_runtime.h>
#include <cmath>

// FlashAttention forward pass. One thread block per (batch, head) pair; within a
// block, thread tx owns exactly one row of the current Q tile — it loads that row,
// computes all Bc dot products against the shared K tile, tracks that row's running
// (m, l) statistics, and writes that row's output. No two threads touch the same row.
//
// K/V tiles are staged in shared memory and reused across all Tr Q tiles, which is
// what keeps the N x N score matrix from ever being materialized.

#define BR 32   // rows per Q tile (also the block size)
#define BC 32   // rows per K/V tile

__global__ void flash_attention_kernel(const float* Q, const float* K, const float* V,
                                       float* O, float* l, float* m,
                                       int H, int N, int D, int Tr, int Tc) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tx = threadIdx.x;

    int qkv_base = (b * H + h) * N * D;
    int lm_base = (b * H + h) * N;

    extern __shared__ float smem[];
    float* Qi = smem;             // Br x D
    float* Kj = Qi + BR * D;      // Bc x D
    float* Vj = Kj + BC * D;      // Bc x D
    float* S = Vj + BC * D;       // Br x Bc

    for (int j = 0; j < Tc; j++) {
        // Every thread loads one row of the K/V tile: coalesced across the warp.
        for (int x = 0; x < D; x++) {
            int kv_idx = qkv_base + (j * BC + tx) * D + x;
            Kj[tx * D + x] = K[kv_idx];
            Vj[tx * D + x] = V[kv_idx];
        }

        __syncthreads(); // all Bc rows of K/V are resident

        for (int i = 0; i < Tr; i++) {
            // Thread tx only ever reads its own Qi slot, so no barrier is needed here.
            for (int x = 0; x < D; x++) {
                Qi[tx * D + x] = Q[qkv_base + (i * BR + tx) * D + x];
            }

            int lm_idx = lm_base + i * BR + tx;
            float l_prev = l[lm_idx];
            float m_prev = m[lm_idx];

            // Scores for this row against all Bc keys in the tile, plus the tile max.
            float row_m = -INFINITY;
            for (int x = 0; x < BC; x++) {
                float prod = 0.0f;
                for (int y = 0; y < D; y++) {
                    prod += Qi[tx * D + y] * Kj[x * D + y];
                }
                prod /= sqrtf((float)D);
                S[tx * BC + x] = prod;
                row_m = fmaxf(row_m, prod);
            }

            // Exponentiate about the tile max and accumulate the tile normalizer.
            float l_ij = 0.0f;
            for (int x = 0; x < BC; x++) {
                float e = __expf(S[tx * BC + x] - row_m);
                S[tx * BC + x] = e;
                l_ij += e;
            }

            // Online softmax merge of (m_prev, l_prev) with this tile's (row_m, l_ij).
            float mi_new = fmaxf(m_prev, row_m);
            float scale_prev = __expf(m_prev - mi_new);
            float scale_cur = __expf(row_m - mi_new);
            float li_new = l_prev * scale_prev + l_ij * scale_cur;

            for (int x = 0; x < D; x++) {
                float pv = 0.0f;
                for (int y = 0; y < BC; y++) {
                    pv += S[tx * BC + y] * Vj[y * D + x];
                }
                int o_idx = qkv_base + (i * BR + tx) * D + x;
                // Rescale the running output and fold in this tile's P*V contribution.
                O[o_idx] = (l_prev * scale_prev * O[o_idx] + scale_cur * pv) / li_new;
            }

            l[lm_idx] = li_new;
            m[lm_idx] = mi_new;
        }

        __syncthreads(); // everyone is done reading K/V before the tile is overwritten
    }
}

// m must start at -inf and l/O at 0, otherwise the first online-softmax merge reads
// garbage (and 0 * NaN would poison the output accumulator).
__global__ void init_stats_kernel(float* l, float* m, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        l[idx] = 0.0f;
        m[idx] = -INFINITY;
    }
}

extern "C" void flash_attention_gpu(const float* d_Q, const float* d_K, const float* d_V,
                                    float* d_O, float* d_l, float* d_m,
                                    int B, int H, int N, int D) {
    int Tr = (N + BR - 1) / BR;
    int Tc = (N + BC - 1) / BC;

    int stats = B * H * N;
    init_stats_kernel<<<(stats + 255) / 256, 256>>>(d_l, d_m, stats);
    cudaMemset(d_O, 0, (size_t)B * H * N * D * sizeof(float));

    dim3 grid(B, H);
    dim3 block(BR);
    size_t smem = (BR * D + 2 * BC * D + BR * BC) * sizeof(float);

    flash_attention_kernel<<<grid, block, smem>>>(d_Q, d_K, d_V, d_O, d_l, d_m,
                                                  H, N, D, Tr, Tc);
}
