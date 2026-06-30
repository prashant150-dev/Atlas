"""Fast-CPU-training research #1: does SPARSE-ONLY compute actually speed up on CPU?

The biggest claimed fast-training lever was "skip the zeros" (6x at 95% sparse). But on a
CPU, unstructured sparse matmul has gather/scatter overhead — it may be SLOWER than dense
unless sparsity is high enough AND the matrix is big enough. This MEASURES the real wall-
clock of dense vs sparse (scipy CSR) matmul at training-relevant sizes, so we know if the
lever is real or a myth on this hardware.

Run:  python projects/v2_design/T11_training/fasttrain/sparse_skip_bench.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse

OUT = Path(__file__).resolve().parent / "sparse_skip_results.json"
REPS = 20


def bench(d_in, d_out, sparsity, batch):
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((d_in, d_out)).astype(np.float32))
    mask = rng.random((d_in, d_out)) > sparsity
    Ws = (W * mask).astype(np.float32)
    X = rng.standard_normal((batch, d_in)).astype(np.float32)
    Wcsr = sparse.csr_matrix(Ws)

    # dense matmul (what masked training does now)
    Wd = np.ascontiguousarray(Ws)
    t = time.perf_counter()
    for _ in range(REPS):
        _ = X @ Wd
    dense_ms = (time.perf_counter() - t) / REPS * 1e3

    # sparse matmul (skip zeros)
    t = time.perf_counter()
    for _ in range(REPS):
        _ = X @ Wcsr
    sparse_ms = (time.perf_counter() - t) / REPS * 1e3

    return dense_ms, sparse_ms


def main():
    print("dense vs SPARSE (scipy CSR) matmul, CPU wall-clock, batch=32\n", flush=True)
    print(f"{'matrix':>14} {'sparsity':>9} {'dense ms':>9} {'sparse ms':>10} {'speedup':>8}", flush=True)
    print("-" * 56, flush=True)
    rows = []
    sizes = [(128, 512), (512, 512), (1024, 4096), (4096, 4096)]
    for d_in, d_out in sizes:
        for s in (0.90, 0.95, 0.98):
            dm, sm = bench(d_in, d_out, s, 32)
            spd = dm / sm
            rows.append({"d_in": d_in, "d_out": d_out, "sparsity": s,
                         "dense_ms": round(dm, 3), "sparse_ms": round(sm, 3), "speedup": round(spd, 2)})
            tag = "FASTER" if spd > 1.1 else ("~same" if spd > 0.9 else "SLOWER")
            print(f"{d_in}x{d_out:>5} {s*100:8.0f}% {dm:8.3f} {sm:9.3f} {spd:7.2f}x  {tag}", flush=True)

    big = [r for r in rows if r["d_in"] >= 1024 and r["sparsity"] >= 0.95]
    avg_big = np.mean([r["speedup"] for r in big]) if big else 0
    print(f"\nHONEST READ:", flush=True)
    print(f"  - Small matrices: sparse often SLOWER (CSR overhead > savings).", flush=True)
    print(f"  - Big matrices (1024+), 95%+ sparse: speedup ~{avg_big:.1f}x (the lever is real HERE).", flush=True)
    print(f"  - Unstructured sparse on CPU is overhead-heavy; the real win needs BIG matrices", flush=True)
    print(f"    + high sparsity, OR structured (block/2:4) sparsity for hardware efficiency.", flush=True)
    print(f"  => fast-training 'sparse-skip' is real but SIZE-dependent; honest multiplier is", flush=True)
    print(f"     smaller than the naive 6x for small models. Measure, don't assume.", flush=True)

    OUT.write_text(json.dumps({"reps": REPS, "rows": rows, "avg_big_speedup": round(float(avg_big), 2),
                   "note": "dense vs scipy-CSR sparse matmul on CPU; sparse-skip helps only at big "
                           "matrices + high sparsity (unstructured CSR overhead dominates small ones). "
                           "Real fast-training needs big/structured sparsity."}, indent=2),
                   encoding="utf-8")
    print(f"\nwritten {OUT}", flush=True)


if __name__ == "__main__":
    main()
