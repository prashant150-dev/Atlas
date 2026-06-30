"""Phase-1 benchmark: our AVX2 packed-2-bit ternary kernel vs numpy fp32.

Honest test of THE speed enabler. Builds a real FFN-sized ternary weight, packs it
to 2 bits, and times:
  - numpy fp32 mat-vec (x @ W_fp32)   — the baseline BLAS path
  - our C kernel on 2-bit packed W    — reads 16x less weight memory, add/sub only
plus a correctness check (kernel must match the dequantized fp32 result).

Run AFTER building libternary.dll:
    bash projects/day14_kernel/build.sh
    python projects/day14_kernel/bench_kernel.py
"""

from __future__ import annotations

import ctypes
import json
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "bench_results.json"
K, N, REPS = 2048, 8192, 50      # one big FFN-ish matrix, single-token decode


def pack_ternary(signs: np.ndarray) -> np.ndarray:
    """signs [K,N] in {-1,0,1} -> packed uint8 [N, ceil(K/4)] column-major (code 0/1/2)."""
    Kd, Nd = signs.shape
    codes = np.zeros_like(signs, dtype=np.uint8)
    codes[signs == 1] = 1
    codes[signs == -1] = 2
    KB = (Kd + 3) // 4
    out = np.zeros((Nd, KB), dtype=np.uint8)
    col = codes.T  # [N,K]
    for j in range(4):
        ks = np.arange(j, Kd, 4)
        out[:, : len(ks)] |= (col[:, ks] << (2 * j)).astype(np.uint8)
    return out


def main():
    lib_path = _HERE / "libternary.dll"
    if not lib_path.exists():
        print("libternary.dll not found — build first:  bash projects/day14_kernel/build.sh")
        return
    lib = ctypes.CDLL(str(lib_path))
    lib.ternary_matvec.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8),
                                   ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                   ctypes.c_int, ctypes.c_int]

    rng = np.random.default_rng(0)
    Wf = rng.standard_normal((K, N)).astype(np.float32)
    thr = 0.7 * np.abs(Wf).mean(0, keepdims=True)
    signs = np.zeros_like(Wf, dtype=np.int8)
    signs[Wf > thr] = 1; signs[Wf < -thr] = -1
    scale = (np.abs(Wf) * (signs != 0)).sum(0) / np.clip((signs != 0).sum(0), 1, None)
    scale = scale.astype(np.float32)
    Wdeq = (signs.astype(np.float32) * scale)        # what the kernel computes
    packed = pack_ternary(signs)
    x = rng.standard_normal(K).astype(np.float32)

    xc = np.ascontiguousarray(x)
    pc = np.ascontiguousarray(packed.reshape(-1))
    sc = np.ascontiguousarray(scale)
    y = np.zeros(N, dtype=np.float32)
    fp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    up = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    # correctness
    lib.ternary_matvec(fp(xc), up(pc), fp(sc), fp(y), K, N)
    y_ref = x @ Wdeq
    err = float(np.max(np.abs(y - y_ref)))

    # timing
    t = time.perf_counter()
    for _ in range(REPS):
        lib.ternary_matvec(fp(xc), up(pc), fp(sc), fp(y), K, N)
    kt = (time.perf_counter() - t) / REPS

    t = time.perf_counter()
    for _ in range(REPS):
        _ = x @ Wdeq
    ft = (time.perf_counter() - t) / REPS

    fp32_mb = Wdeq.nbytes / 1e6
    pk_mb = (pc.nbytes + sc.nbytes) / 1e6
    print(f"matrix K={K} N={N} (one decode step)")
    print(f"  weight RAM: fp32 {fp32_mb:.1f} MB  ->  packed {pk_mb:.2f} MB  ({fp32_mb/pk_mb:.1f}x less)")
    print(f"  correctness max|err| = {err:.3e}")
    print(f"  numpy fp32 : {ft*1e6:8.1f} us")
    print(f"  our kernel : {kt*1e6:8.1f} us   ({ft/kt:.2f}x vs fp32)")
    OUT.write_text(json.dumps({"K": K, "N": N, "fp32_us": round(ft*1e6, 1),
                               "kernel_us": round(kt*1e6, 1), "speedup_x": round(ft/kt, 3),
                               "fp32_MB": round(fp32_mb, 1), "packed_MB": round(pk_mb, 2),
                               "ram_reduction_x": round(fp32_mb/pk_mb, 1),
                               "max_err": err}, indent=2), encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
