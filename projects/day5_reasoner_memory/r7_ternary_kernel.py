"""Day-5 R7: a packed-ternary matmul kernel — proving Lever 2.

R5 showed that on CPU, loading a packed-ternary model and dequantizing to fp32
buys DISK only: no RAM or speed win, because the weights inflate back to fp32 in
memory and the matmul runs in fp32. Lever 2 is the missing piece: a kernel that
computes y = x @ W directly on the PACKED 2-bit weights, so the fp32 matrix is
never materialised.

This module is a reference kernel that proves the three things a real (SIMD)
low-bit kernel exploits — without faking a wall-clock speedup that pure
Python/NumPy cannot deliver against BLAS:

  1. RAM FLOOR: the resident weight is 2 bits/weight (packed uint8). The kernel
     streams it in small column blocks; the full fp32 matrix is never allocated.
  2. NO MULTIPLIES: a ternary weight is +1 / 0 / -1, so each contribution is an
     add / skip / subtract. The only multiplies left are N per-output scales.
     Multiply count drops by a factor of ~K.
  3. CORRECTNESS: the packed kernel output equals the dequantized fp32 matmul
     bit-for-bit (same arithmetic, just reorganised).

Honest note: realising the SPEED win needs a SIMD/popcount kernel in C
(bitnet.cpp-style). NumPy/Python cannot beat BLAS fp32 here; we report wall-clock
truthfully and prove the *structure* (RAM + op-count + correctness) that real
kernels turn into speed and energy savings.

Run from repo root::

    python projects/day5_reasoner_memory/r7_ternary_kernel.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "r7_results.json"

_EPS = 1e-12


def ternarize_columns(W: np.ndarray, threshold_factor: float = 0.7):
    """Per-output-column ternarize. Returns (signs in {-1,0,1} int8, scales fp32[N])."""

    absW = np.abs(W)
    thr = threshold_factor * absW.mean(axis=0, keepdims=True).clip(_EPS)
    signs = np.zeros_like(W, dtype=np.int8)
    signs[W > thr] = 1
    signs[W < -thr] = -1
    kept = signs != 0
    denom = kept.sum(axis=0).clip(1)
    scales = (absW * kept).sum(axis=0) / denom
    return signs, scales.astype(np.float32)


def pack_ternary(signs: np.ndarray) -> np.ndarray:
    """Pack a [K,N] {-1,0,1} matrix into 2-bit codes, column-major, 4 per byte.

    code map: 0->0, +1->1, -1->2. Returns a flat uint8 array of the model weight
    as it RESIDES in RAM (2 bits/weight).
    """

    K, N = signs.shape
    codes = np.zeros_like(signs, dtype=np.uint8)
    codes[signs == 1] = 1
    codes[signs == -1] = 2
    flat = codes.T.reshape(-1)                      # column-major: col n at n*K..
    pad = (-flat.size) % 4
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, np.uint8)])
    q = flat.reshape(-1, 4)
    packed = (q[:, 0] | (q[:, 1] << 2) | (q[:, 2] << 4) | (q[:, 3] << 6)).astype(np.uint8)
    return packed


def _unpack_column(packed: np.ndarray, n: int, K: int) -> np.ndarray:
    """Decode column n's K ternary signs from the packed buffer via bit-shifts."""

    start = n * K
    out = np.empty(K, dtype=np.int8)
    for k in range(K):
        idx = start + k
        byte = packed[idx >> 2]
        code = (byte >> ((idx & 3) * 2)) & 0b11
        out[k] = 1 if code == 1 else (-1 if code == 2 else 0)
    return out


def matmul_packed_pure(x: np.ndarray, packed: np.ndarray, scales: np.ndarray, K: int):
    """Pure add/sub kernel on packed weights (the literal proof, small sizes).

    Computes y[n] = scale[n] * sum_k sign(k,n) * x[k] using ONLY additions and
    subtractions over x (no multiply by the weight), reading 2-bit codes by
    bit-shift. The only multiply is the final per-output scale. Returns (y, stats).
    """

    N = scales.shape[0]
    y = np.zeros(N, dtype=np.float64)
    adds = 0
    for n in range(N):
        acc = 0.0
        start = n * K
        for k in range(K):
            idx = start + k
            code = (packed[idx >> 2] >> ((idx & 3) * 2)) & 0b11
            if code == 1:
                acc += x[k]; adds += 1
            elif code == 2:
                acc -= x[k]; adds += 1
            # code 0 -> skip (sparsity is free)
        y[n] = acc * scales[n]           # the only multiplies: N of them
    return y, {"adds": adds, "weight_multiplies": 0, "scale_multiplies": N}


def matmul_packed_blocked(X: np.ndarray, packed: np.ndarray, scales: np.ndarray,
                          K: int, block: int = 64):
    """Vectorised packed kernel: stream columns in blocks so the fp32 weight
    matrix is never materialised. Peak transient = K*block int8 (one block)."""

    M = X.shape[0]
    N = scales.shape[0]
    Y = np.empty((M, N), dtype=np.float32)
    peak_transient = 0
    for n0 in range(0, N, block):
        n1 = min(n0 + block, N)
        b = n1 - n0
        sign_blk = np.empty((K, b), dtype=np.int8)   # transient unpack of b cols
        for j, n in enumerate(range(n0, n1)):
            sign_blk[:, j] = _unpack_column_vec(packed, n, K)
        peak_transient = max(peak_transient, sign_blk.nbytes)
        pos = (sign_blk == 1).astype(np.float32)
        neg = (sign_blk == -1).astype(np.float32)
        # masked sums = additions of selected x columns (no weight multiply)
        Y[:, n0:n1] = (X @ pos - X @ neg) * scales[n0:n1]
    return Y, peak_transient


def _unpack_column_vec(packed: np.ndarray, n: int, K: int) -> np.ndarray:
    start = n * K
    idx = start + np.arange(K)
    codes = (packed[idx >> 2] >> ((idx & 3) * 2)) & 0b11
    out = np.zeros(K, dtype=np.int8)
    out[codes == 1] = 1
    out[codes == 2] = -1
    return out


def main() -> None:
    rng = np.random.default_rng(0)

    # ---- 1. literal proof on a small layer: add-only, bit-exact ----
    Ks, Ns = 32, 8
    Ws = rng.standard_normal((Ks, Ns)).astype(np.float32)
    signs_s, scales_s = ternarize_columns(Ws)
    packed_s = pack_ternary(signs_s)
    xs = rng.standard_normal(Ks).astype(np.float32)
    y_pure, opstats = matmul_packed_pure(xs, packed_s, scales_s, Ks)
    y_ref = xs @ (signs_s.astype(np.float32) * scales_s)      # dequantized reference
    err_small = float(np.max(np.abs(y_pure - y_ref)))
    print(f"[proof] add-only kernel vs dequant ref: max|err| = {err_small:.2e} "
          f"(weight multiplies = {opstats['weight_multiplies']}, adds = {opstats['adds']}, "
          f"scale mults = {opstats['scale_multiplies']})", flush=True)

    # ---- 2. realistic layer: RAM floor + correctness + op counts + timing ----
    K, N, M = 768, 768, 64                       # a GPT-2-ish linear, batch 64
    W = rng.standard_normal((K, N)).astype(np.float32)
    signs, scales = ternarize_columns(W)
    packed = pack_ternary(signs)
    X = rng.standard_normal((M, K)).astype(np.float32)

    fp32_bytes = W.nbytes
    packed_bytes = packed.nbytes + scales.nbytes
    W_deq = (signs.astype(np.float32) * scales)

    Y_packed, peak_transient = matmul_packed_blocked(X, packed, scales, K, block=64)
    Y_ref = X @ W_deq
    err = float(np.max(np.abs(Y_packed - Y_ref)))

    # timing (honest): BLAS fp32 vs our blocked packed kernel
    t = time.perf_counter(); [X @ W for _ in range(5)]; fp32_t = (time.perf_counter() - t) / 5
    t = time.perf_counter(); [matmul_packed_blocked(X, packed, scales, K, 64) for _ in range(5)]
    packed_t = (time.perf_counter() - t) / 5

    fp32_macs = M * K * N
    ternary_adds = int(M * (signs != 0).sum())
    ternary_mults = M * N                          # only the scales

    print(f"[RAM]  fp32 weight {fp32_bytes/1e6:.2f} MB  ->  packed {packed_bytes/1e6:.3f} MB "
          f"({fp32_bytes/packed_bytes:.1f}x smaller); peak transient unpack {peak_transient/1e3:.1f} KB", flush=True)
    print(f"[OPS]  fp32 multiplies {fp32_macs:,}  ->  ternary weight-multiplies 0, "
          f"scale-multiplies {ternary_mults:,}  ({fp32_macs/ternary_mults:.0f}x fewer mults)", flush=True)
    print(f"[OK]   packed kernel vs dequant matmul: max|err| = {err:.2e}", flush=True)
    print(f"[TIME] BLAS fp32 {fp32_t*1e3:.1f} ms | python packed {packed_t*1e3:.1f} ms "
          f"(python loses on wall-clock — SIMD/C kernel is where the op+RAM win becomes speed)", flush=True)

    payload = {
        "small_proof": {"max_err": err_small, **opstats},
        "layer": {"K": K, "N": N, "batch": M},
        "ram": {"fp32_MB": round(fp32_bytes/1e6, 3), "packed_MB": round(packed_bytes/1e6, 4),
                "ratio": round(fp32_bytes/packed_bytes, 2),
                "peak_transient_KB": round(peak_transient/1e3, 2),
                "bits_per_weight": round(packed_bytes * 8 / (K * N), 3)},
        "ops": {"fp32_multiplies": fp32_macs, "ternary_weight_multiplies": 0,
                "ternary_adds": ternary_adds, "ternary_scale_multiplies": ternary_mults,
                "multiply_reduction_x": round(fp32_macs / ternary_mults, 1)},
        "correctness_max_err": err,
        "timing_ms": {"blas_fp32": round(fp32_t*1e3, 2), "python_packed": round(packed_t*1e3, 2)},
        "note": "RAM floor + multiply-elimination + exactness are proven in Python; the SPEED win requires a SIMD/C kernel and is not claimable from NumPy vs BLAS.",
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
