"""Phase-1 kernel via Numba (LLVM JIT, no system C compiler needed).

Packed 2-bit ternary mat-vec on this Haswell CPU. The honest test: does computing
directly on 2-bit-packed weights (16x less weight RAM, add/sub only) beat numpy
fp32? LLVM auto-vectorises the parallel reduction to AVX2.

Run:  python projects/day14_kernel/numba_kernel.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "numba_results.json"
K, N, REPS = 2048, 8192, 50


_LUT = np.array([0.0, 1.0, -1.0, 0.0], dtype=np.float32)   # code 0/1/2/3 -> sign


@njit(parallel=True, fastmath=True, cache=True)
def ternary_matvec(x, packed, scale, y, K, N, KB, lut):
    """y[n] = scale[n] * sum_k sign(W[k,n]) * x[k]; packed = [N*KB] uint8, 2-bit codes.
    K % 4 == 0; sign via 4-entry LUT (one load+mul per weight)."""
    for n in prange(N):
        base = n * KB
        acc = np.float32(0.0)
        for b in range(KB):
            byte = packed[base + b]
            k = b * 4
            acc += lut[byte & 3] * x[k]
            acc += lut[(byte >> 2) & 3] * x[k + 1]
            acc += lut[(byte >> 4) & 3] * x[k + 2]
            acc += lut[(byte >> 6) & 3] * x[k + 3]
        y[n] = acc * scale[n]


def pack(signs):
    Kd, Nd = signs.shape
    codes = np.zeros_like(signs, dtype=np.uint8)
    codes[signs == 1] = 1
    codes[signs == -1] = 2
    KB = (Kd + 3) // 4
    out = np.zeros((Nd, KB), dtype=np.uint8)
    col = codes.T
    for j in range(4):
        ks = np.arange(j, Kd, 4)
        out[:, : len(ks)] |= (col[:, ks].astype(np.uint8) << (2 * j))
    return out, KB


def main():
    rng = np.random.default_rng(0)
    Wf = rng.standard_normal((K, N)).astype(np.float32)
    thr = 0.7 * np.abs(Wf).mean(0, keepdims=True)
    signs = np.zeros_like(Wf, dtype=np.int8)
    signs[Wf > thr] = 1; signs[Wf < -thr] = -1
    scale = ((np.abs(Wf) * (signs != 0)).sum(0) / np.clip((signs != 0).sum(0), 1, None)).astype(np.float32)
    Wdeq = signs.astype(np.float32) * scale
    packed, KB = pack(signs)
    pflat = np.ascontiguousarray(packed.reshape(-1))
    x = rng.standard_normal(K).astype(np.float32)
    y = np.zeros(N, dtype=np.float32)

    ternary_matvec(x, pflat, scale, y, K, N, KB, _LUT)        # warm-up (JIT compile)
    y_ref = x @ Wdeq
    err = float(np.max(np.abs(y - y_ref)))

    t = time.perf_counter()
    for _ in range(REPS):
        ternary_matvec(x, pflat, scale, y, K, N, KB, _LUT)
    kt = (time.perf_counter() - t) / REPS

    t = time.perf_counter()
    for _ in range(REPS):
        _ = x @ Wdeq
    ft = (time.perf_counter() - t) / REPS

    fp32_mb = Wdeq.nbytes / 1e6
    pk_mb = (pflat.nbytes + scale.nbytes) / 1e6
    print(f"matrix K={K} N={N} (single-token decode), Numba LLVM kernel")
    print(f"  weight RAM : fp32 {fp32_mb:.1f} MB -> packed {pk_mb:.2f} MB  ({fp32_mb/pk_mb:.1f}x less)")
    print(f"  correctness: max|err| = {err:.3e}")
    print(f"  numpy fp32 : {ft*1e6:8.1f} us")
    print(f"  our kernel : {kt*1e6:8.1f} us   ({ft/kt:.2f}x vs fp32)")
    OUT.write_text(json.dumps({"K": K, "N": N, "fp32_us": round(ft*1e6, 1),
                               "kernel_us": round(kt*1e6, 1), "speedup_x": round(ft/kt, 3),
                               "fp32_MB": round(fp32_mb, 1), "packed_MB": round(pk_mb, 2),
                               "ram_reduction_x": round(fp32_mb/pk_mb, 1), "max_err": err},
                              indent=2), encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
