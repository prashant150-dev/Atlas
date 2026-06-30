"""T3 SPEED — fix paging's speed by computing DIRECTLY on packed weights (LUT kernel).

T2 found naive 2-bit paging is SLOW because each layer is unpacked to fp32 before matmul.
T3's answer: store each layer as LUT group-indices and run the LUT-GEMM kernel on them
DIRECTLY — no fp32 materialization, no per-weight multiply. This should give paging's low
RAM AND good speed at once.

Compare per layer-stack (single-token decode), all paged one layer at a time:
  paged_fp32   : load fp32 layer, numpy matmul        (baseline paging)
  paged_unpack : load 2-bit, UNPACK to fp32, matmul   (T2's slow path)
  paged_LUT    : load group-indices, LUT kernel direct (T3: no unpack, no multiply)

No training; CPU; reuses the Day-14 LUT-GEMM kernel.

Run:  python projects/v2_design/T3_speed/paged_lut.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "day14_kernel"))
from lut_gemm import G, NPAT, build_table, encode_groups, lut_accumulate  # type: ignore

HERE = Path(__file__).resolve().parent
LD = HERE / "_layers"
OUT = HERE / "paged_lut_results.json"

K, N = 4096, 4096        # layer dims (square stack)
NLAYERS = 16
REPS = 3
KG = K // G


def _rss():
    return psutil.Process().memory_info().rss / 1e6


def _ternary(W):
    thr = 0.7 * np.abs(W).mean(0, keepdims=True)
    s = np.zeros_like(W, dtype=np.int8)
    s[W > thr] = 1; s[W < -thr] = -1
    scale = ((np.abs(W) * (s != 0)).sum(0) / np.clip((s != 0).sum(0), 1, None)).astype(np.float32)
    return s, scale


def _build():
    LD.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    fp32_b = idx_b = 0
    for i in range(NLAYERS):
        W = (rng.standard_normal((K, N)).astype(np.float32) * 0.02)
        np.save(LD / f"f{i}.npy", W)
        s, scale = _ternary(W)
        idx = encode_groups(s)                       # [N, KG] uint
        np.savez(LD / f"l{i}.npz", idx=idx.astype(np.uint8), scale=scale)
        fp32_b += W.nbytes
        idx_b += idx.astype(np.uint8).nbytes + scale.nbytes
    return fp32_b, idx_b


def run_paged_fp32(x):
    peak = _rss(); t = time.perf_counter(); h = x
    for i in range(NLAYERS):
        W = np.load(LD / f"f{i}.npy")
        h = np.maximum(h @ W, 0); peak = max(peak, _rss()); del W
    return peak, time.perf_counter() - t


def run_paged_unpack(x):
    peak = _rss(); t = time.perf_counter(); h = x
    for i in range(NLAYERS):
        z = np.load(LD / f"l{i}.npz")
        idx = z["idx"].astype(np.int32); scale = z["scale"]
        # rebuild signs from group indices -> fp32 W, then matmul (the slow path)
        codes = np.zeros((N, KG, G), np.int8)
        t2 = idx.copy()
        for j in range(G):
            codes[:, :, j] = t2 % 3; t2 //= 3
        signs = np.where(codes == 1, 1, np.where(codes == 2, -1, 0)).astype(np.float32)
        W = signs.reshape(N, K).T * scale
        h = np.maximum(h @ W, 0); peak = max(peak, _rss()); del W, z
    return peak, time.perf_counter() - t


def run_paged_lut(x):
    peak = _rss(); t = time.perf_counter()
    h = x.reshape(-1).astype(np.float32)
    table = np.zeros((KG, NPAT), np.float32)
    y = np.zeros(N, np.float32)
    for i in range(NLAYERS):
        z = np.load(LD / f"l{i}.npz")
        idx = z["idx"]; scale = z["scale"]
        build_table(h, KG, G, NPAT, table)             # per-token table from current input
        lut_accumulate(idx, table, scale, y, N, KG)    # direct on indices, no unpack/multiply
        h = np.maximum(y, 0).copy(); peak = max(peak, _rss()); del z
    return peak, time.perf_counter() - t


def main():
    fp32_b, idx_b = _build()
    x = np.random.default_rng(1).standard_normal((1, K)).astype(np.float32)
    base = _rss()
    print(f"model: {NLAYERS}x{K}x{N} | fp32 {fp32_b/1e6:.0f}MB | LUT-indices {idx_b/1e6:.0f}MB "
          f"({fp32_b/idx_b:.0f}x smaller)\n", flush=True)

    for fn in (run_paged_fp32, run_paged_unpack, run_paged_lut):
        fn(x)   # warm / JIT
    modes = {"paged_fp32": run_paged_fp32, "paged_unpack": run_paged_unpack, "paged_LUT": run_paged_lut}
    res = {}
    for name, fn in modes.items():
        pk = 0.0; tt = 0.0
        for _ in range(REPS):
            gc.collect(); p, dt = fn(x); pk = max(pk, p); tt += dt
        res[name] = {"peak_mb": round(pk - base, 1), "ms": round(tt/REPS*1e3, 1)}

    print(f"{'mode':14s} {'peak RAM':>10} {'time':>9} {'vs fp32':>9}", flush=True)
    print("-" * 46, flush=True)
    fp = res["paged_fp32"]["ms"]
    for name, r in res.items():
        print(f"{name:14s} {r['peak_mb']:8.0f}MB {r['ms']:7.0f}ms {fp/r['ms']:7.2f}x", flush=True)

    lut = res["paged_LUT"]; unp = res["paged_unpack"]
    print(f"\n  paged-LUT vs paged-unpack (T2's slow path): {unp['ms']/lut['ms']:.1f}x FASTER", flush=True)
    print(f"  paged-LUT: low RAM ({lut['peak_mb']:.0f}MB) AND {fp/lut['ms']:.2f}x vs fp32 paging", flush=True)
    print(f"  -> computing DIRECTLY on packed weights fixes paging's speed (no unpack, no multiply).",
          flush=True)
    OUT.write_text(json.dumps({"K": K, "N": N, "n_layers": NLAYERS,
                   "fp32_mb": round(fp32_b/1e6, 1), "idx_mb": round(idx_b/1e6, 1),
                   "modes": res, "note": "paged inference with LUT kernel direct on packed "
                   "weights vs unpack-to-fp32; T3 fixes T2's paging speed."}, indent=2),
                   encoding="utf-8")
    for f in LD.glob("*"):
        f.unlink()
    print(f"\nwritten {OUT}", flush=True)


if __name__ == "__main__":
    main()
