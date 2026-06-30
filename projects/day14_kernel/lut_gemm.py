"""Phase-1b: LUT-GEMM ternary kernel (T-MAC style) — beat fp32 by removing multiplies.

Idea: split x into groups of g elements. For each group there are 3^g possible
ternary weight patterns; precompute the partial sum  sum_j sign_j * x[j]  for ALL
of them into a table T (once per token, O(N_groups * 3^g)). Then each output column
just LOOKS UP its precomputed partial sum per group and ADDS — no per-weight
multiply, and the weight is read as a tiny group-index.

Cost model (single token, K inputs, N outputs, group g):
  table build : (K/g) * 3^g   multiply-adds   (shared across ALL N outputs)
  accumulate  : N * (K/g)      table-lookups + adds   (no multiplies)
vs dense fp32 : N * K          multiply-adds
So if N is large, the per-output work drops by ~g (only K/g adds, no mults), and the
table build amortises. g=4 -> 3^4=81 table entries/group, K/4 adds per output.

We pack each group's ternary pattern into one index in [0, 3^g). Weight storage is
ceil(log2(3^g)) bits/group ~= 1.585*g bits / g = 1.585 bits/weight (true ternary).

Run:  python projects/day14_kernel/lut_gemm.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "lut_results.json"
K, N, REPS = 2048, 8192, 50
G = 4                       # group size; 3^4 = 81 patterns
NPAT = 3 ** G


def encode_groups(signs):
    """signs [K,N] in {-1,0,1} -> group indices [N, K/G] in [0,3^G).
    code per element: 0->0, +1->1, -1->2; index = sum code_j * 3^j."""
    Kd, Nd = signs.shape
    assert Kd % G == 0
    codes = np.zeros_like(signs, dtype=np.int32)
    codes[signs == 1] = 1
    codes[signs == -1] = 2
    col = codes.T.reshape(Nd, Kd // G, G)             # [N, KG, G]
    powers = (3 ** np.arange(G)).astype(np.int32)
    idx = (col * powers[None, None, :]).sum(-1).astype(np.int32)   # [N, KG]
    return np.ascontiguousarray(idx)


@njit(cache=True)
def build_table(x, KG, G, NPAT, table):
    """table[gi, pat] = sum_j sign(pat_j) * x[gi*G + j]  for all patterns pat."""
    for gi in range(KG):
        base = gi * G
        for pat in range(NPAT):
            p = pat
            acc = np.float32(0.0)
            for j in range(G):
                c = p % 3
                p //= 3
                if c == 1:
                    acc += x[base + j]
                elif c == 2:
                    acc -= x[base + j]
            table[gi, pat] = acc


@njit(parallel=True, fastmath=True, cache=True)
def lut_accumulate(idx, table, scale, y, N, KG):
    """y[n] = scale[n] * sum_gi table[gi, idx[n,gi]]  — lookups + adds, no multiplies."""
    for n in prange(N):
        acc = np.float32(0.0)
        for gi in range(KG):
            acc += table[gi, idx[n, gi]]
        y[n] = acc * scale[n]


def main():
    rng = np.random.default_rng(0)
    Wf = rng.standard_normal((K, N)).astype(np.float32)
    thr = 0.7 * np.abs(Wf).mean(0, keepdims=True)
    signs = np.zeros_like(Wf, dtype=np.int8)
    signs[Wf > thr] = 1; signs[Wf < -thr] = -1
    scale = ((np.abs(Wf) * (signs != 0)).sum(0) / np.clip((signs != 0).sum(0), 1, None)).astype(np.float32)
    Wdeq = signs.astype(np.float32) * scale
    idx = encode_groups(signs)                    # [N, KG]
    KG = K // G
    x = rng.standard_normal(K).astype(np.float32)
    table = np.zeros((KG, NPAT), dtype=np.float32)
    y = np.zeros(N, dtype=np.float32)

    # warm-up (JIT)
    build_table(x, KG, G, NPAT, table)
    lut_accumulate(idx, table, scale, y, N, KG)
    err = float(np.max(np.abs(y - x @ Wdeq)))

    t = time.perf_counter()
    for _ in range(REPS):
        build_table(x, KG, G, NPAT, table)
        lut_accumulate(idx, table, scale, y, N, KG)
    lt = (time.perf_counter() - t) / REPS

    t = time.perf_counter()
    for _ in range(REPS):
        _ = x @ Wdeq
    ft = (time.perf_counter() - t) / REPS

    # weight storage: index in [0,81) -> 7 bits/group / 4 = 1.75 b/w (or pack 81 as ~6.34b)
    bpw = np.log2(NPAT) / G
    fp32_mb = Wdeq.nbytes / 1e6
    idx_mb = idx.astype(np.uint8).nbytes / 1e6 + scale.nbytes / 1e6
    print(f"LUT-GEMM ternary  K={K} N={N} g={G} (3^{G}={NPAT} patterns)")
    print(f"  weight storage : {bpw:.2f} bits/weight ({fp32_mb:.1f} MB fp32 -> {idx_mb:.2f} MB idx)")
    print(f"  correctness    : max|err| = {err:.3e}")
    print(f"  numpy fp32     : {ft*1e6:8.1f} us")
    print(f"  LUT-GEMM kernel: {lt*1e6:8.1f} us   ({ft/lt:.2f}x vs fp32)")
    OUT.write_text(json.dumps({"K": K, "N": N, "G": G, "patterns": NPAT,
                               "bits_per_weight": round(bpw, 3),
                               "fp32_us": round(ft*1e6, 1), "lut_us": round(lt*1e6, 1),
                               "speedup_x": round(ft/lt, 3), "max_err": err}, indent=2),
                   encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
