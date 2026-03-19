# CUDA Kernels — SysML

CUDA implementations for two systems-for-ML assignments: a progressively optimized
matrix multiplication (PA2) and a two-layer MLP forward pass (PA3). The assignment
statements are the PDFs in this directory.

## PA2 — Optimizing matrix multiplication

Ten implementations of `C = A * B`, each a self-contained kernel behind a common
`matmul_gpu(A, B, C, M, N, K)` interface, taking the same problem from a naive
one-thread-per-output kernel all the way to a double-buffered, vectorized,
register-tiled kernel benchmarked against cuBLAS.

```
cd pa2
make            # builds and runs every kernel in src/ against the shared harness
```

`main.cc` fills `A` and `B` with random floats, checks each kernel against a naive CPU
triple-loop, times it with CUDA events, and prints a `cublasSgemm` reference time on the
same inputs — so every variant is validated and measured under identical conditions.

| File | Technique |
|---|---|
| `src/1_matmul_no_coal.cu` | Naive kernel — one thread per output element, uncoalesced global reads |
| `src/2_matmul_coal.cu` | Row/column indexing swapped so a warp reads contiguous memory |
| `src/3_tile_basic.cu` | Shared-memory tiling — one thread loads 1 element of A and 1 of B, computes 1 output |
| `src/4_tiling_row.cu` | 1D tiling — one thread loads 1 element of A and 8 strided elements of B, computes 8 outputs along a row |
| `src/5_tiling_col.cu` | 1D tiling — one thread loads 8 contiguous elements of A and 1 of B, computes 8 outputs along a column |
| `src/6_tiling_2d.cu` | 2D shared-memory + register tiling — 64x64 block tile, each thread accumulates a 4x4 sub-tile of C in registers |
| `src/7_register_tiling.cu` | Same blocking with A held entirely in registers, halving the shared-memory footprint |
| `src/8_vectorized.cu` | 128x128 tiles with an 8x8 register block, `float4` global loads, A transposed in shared memory |
| `src/9_double_buffered.cu` | Two alternating tile buffers — tile `t+1` loads while tile `t` computes, one `__syncthreads` per tile instead of two |
| `src/10_cublas.cu` | `cublasSgemm` behind the same interface, giving a library baseline through the identical harness |

Together these cover all three parts of the assignment: Part A's CPU / naive / coalesced /
cuBLAS comparison, Part B's three required tiling strategies, and Part C's register
tiling and shared-memory-minimizing optimizations.

**The optimization story.** Each variant removes the bottleneck the previous one hit.
Coalescing fixes the memory access pattern; shared-memory tiling cuts global traffic by
the tile width; 1D and then 2D register tiling raise arithmetic intensity so each loaded
value feeds many FMAs; vectorized `float4` loads and a transposed A tile fix load width
and shared-memory bank conflicts; double buffering hides the remaining global-load
latency behind compute.

Both `TILE_SIZE`/`NELEM` (variants 3–5) and `BM`/`BN`/`BK`/`TM`/`TN` (variants 6–9) are
compile-time constants at the top of each file, so tile shapes can be swept to study
their effect on occupancy and runtime.

## PA3 — Two-layer MLP forward pass

Computes `Y = ReLU(X * W1)` then `Z = Y * W2` for a batch of `B = 32` inputs of dimension
`N`, with matmul and ReLU as separate kernels as the assignment requires.

```
cd pa3
make            # builds and runs each forward-pass variant over the full N sweep
```

| File | Contents |
|---|---|
| `src/kernels.cuh` | Shared declarations for the kernels and the `ffnn_forward_gpu` entry point |
| `src/matmul.cu` | Tiled matmul kernel, retuned for the MLP's short 32-row activation matrices |
| `src/relu.cu` | Elementwise ReLU kernel |
| `src/ffnn_forward_normal.cu` | Batched forward pass — matmul → ReLU → matmul in the default stream |
| `src/ffnn_forward_normal_streams.cu` | Stream-parallel variant — the batch is split into 4 chunks of `B/4` rows, each running its own dependent kernel chain in its own CUDA stream so the chunks overlap |
| `src/ffnn_forward_cublas.cu` | Both GEMMs handed to `cublasSgemm`, as a ceiling to measure the hand-written kernels against |

`main.cc` sweeps `N` from 32 to 32K in powers of two, times each variant with the CUDA
events API after a warmup iteration, averages over 10 runs, reports achieved GFLOP/s, and
checks the small sizes against a CPU reference forward pass. Sizes whose weight matrices
would not fit in device memory are skipped rather than failing, so the same binary runs
across GPUs of different capacities.

## Environment

Built with `nvcc` against `sm_70`+ and linked with cuBLAS; developed and benchmarked on
a Kaggle GPU runtime.
