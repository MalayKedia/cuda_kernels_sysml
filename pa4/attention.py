"""Parts A-C: standard attention, the FlashAttention algorithm in Python loops, and
PyTorch's built-in fused kernel, timed against each other over a sweep of sequence
lengths.

Part D (the CUDA kernel) lives in src/flash_attention.cu and is built with the Makefile.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

B = 8
N_H = 16
D = 64

TIMED_ITERS = 10
WARMUP_ITERS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Part A: standard scaled dot-product attention -------------------------------
# Materializes the full N x N score matrix. Correctness baseline for everything else.

def standard_attention(Q, K, V):
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V)


# --- Part B: FlashAttention in Python loops --------------------------------------
# Never forms the N x N matrix: K/V are walked in Bc-row tiles, Q in Br-row tiles, and
# each Q tile keeps running (m, l) statistics that are rescaled as new tiles arrive.
# Deliberately slow -- the point is the online-softmax update, not throughput.

def flash_attention_loops(Q, K, V):
    N = Q.shape[2]

    M = 32 * 1024                  # on-chip SRAM budget, in floats
    Bc = M // (4 * D)
    Br = min(Bc, D)
    Tc = (N + Bc - 1) // Bc
    Tr = (N + Br - 1) // Br

    O = torch.zeros_like(Q, dtype=torch.float32)
    l = torch.zeros((B, N_H, N, 1), device=device, dtype=torch.float32)
    m = torch.full((B, N_H, N, 1), float("-inf"), device=device, dtype=torch.float32)

    for j in range(Tc):
        K_j = K[:, :, j * Bc:(j + 1) * Bc, :].float()
        V_j = V[:, :, j * Bc:(j + 1) * Bc, :].float()

        for i in range(Tr):
            rows = slice(i * Br, (i + 1) * Br)

            Q_i = Q[:, :, rows, :].float()
            O_i = O[:, :, rows, :]
            l_i = l[:, :, rows, :]
            m_i = m[:, :, rows, :]

            S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) / (D ** 0.5)
            m_ij = torch.max(S_ij, dim=-1, keepdim=True).values
            P_ij = torch.exp(S_ij - m_ij)
            l_ij = torch.sum(P_ij, dim=-1, keepdim=True)

            m_new = torch.maximum(m_i, m_ij)
            scale_prev = torch.exp(m_i - m_new)
            scale_cur = torch.exp(m_ij - m_new)
            l_new = scale_prev * l_i + scale_cur * l_ij

            # Rescale the running output, then fold in this tile's P @ V.
            O[:, :, rows, :] = (l_i * scale_prev * O_i + scale_cur * torch.matmul(P_ij, V_j)) / l_new

            l[:, :, rows, :] = l_new
            m[:, :, rows, :] = m_new

    return O.half()


# --- Part C: PyTorch's built-in fused attention ----------------------------------

def flash_attention_builtin(Q, K, V):
    return torch.nn.functional.scaled_dot_product_attention(Q, K, V)


def benchmark(fn, Q, K, V):
    for _ in range(WARMUP_ITERS):
        fn(Q, K, V)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(TIMED_ITERS):
        out = fn(Q, K, V)
    end.record()
    torch.cuda.synchronize()

    return out, start.elapsed_time(end) / TIMED_ITERS


if __name__ == "__main__":
    seq_lens = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    results = {"standard": [], "flash_loops": [], "builtin": []}

    print(f"{'N':>7}{'standard':>13}{'flash(loops)':>15}{'builtin':>12}{'max diff':>14}")
    print("-" * 62)

    for N in seq_lens:
        Q = torch.randn(B, N_H, N, D, device=device, dtype=torch.float16)
        K = torch.randn(B, N_H, N, D, device=device, dtype=torch.float16)
        V = torch.randn(B, N_H, N, D, device=device, dtype=torch.float16)

        out_std, t_std = benchmark(standard_attention, Q, K, V)
        out_flash, t_flash = benchmark(flash_attention_loops, Q, K, V)
        out_builtin, t_builtin = benchmark(flash_attention_builtin, Q, K, V)

        # All three must agree to the handout's 1e-3 bar.
        diff = max(
            (out_std.float() - out_flash.float()).abs().max().item(),
            (out_std.float() - out_builtin.float()).abs().max().item(),
        )

        results["standard"].append(t_std)
        results["flash_loops"].append(t_flash)
        results["builtin"].append(t_builtin)

        flag = "" if diff < 1e-3 else "  <-- exceeds 1e-3"
        print(f"{N:>7}{t_std:>11.3f}ms{t_flash:>13.3f}ms{t_builtin:>10.3f}ms{diff:>14.2e}{flag}")

    plt.figure(figsize=(7, 5))
    plt.plot(seq_lens, results["standard"], marker="o", label="Part A: standard attention")
    plt.plot(seq_lens, results["flash_loops"], marker="s", label="Part B: FlashAttention (Python loops)")
    plt.plot(seq_lens, results["builtin"], marker="^", label="Part C: PyTorch built-in")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("sequence length N")
    plt.ylabel("runtime (ms)")
    plt.title(f"Attention runtime vs N  (B={B}, heads={N_H}, D={D})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("attention_times.png", dpi=150)
    print("\nwrote attention_times.png")
