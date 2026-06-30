"""T2 MEMORY — run a model in FAR less RAM than its size, by paging layers from disk.

Current AI loads the WHOLE model into RAM (70B fp16 = 140GB RAM -> won't fit a PC). T2's
lever: keep only ONE layer (or expert) resident at a time; stream the rest from disk
just-in-time. Peak RAM = one layer, not the whole model. (This is the AirLLM / MoE idea.)
No training needed — pure runtime engineering, CPU-friendly.

This measures REAL process RAM (psutil RSS) for two ways of running the same stacked-layer
"model": load-all vs streamed. Honest tradeoff: streaming saves RAM but costs disk-read
time (T2 vs T3 pull against each other).

Run:  python projects/v2_design/T2_memory/paged_inference.py
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import psutil

HERE = Path(__file__).resolve().parent
LAYERS_DIR = HERE / "_layers"
OUT = HERE / "paged_results.json"

D = 4096          # layer width (real-model-ish)
NLAYERS = 24      # stacked layers
REPS = 3


def _rss_mb():
    return psutil.Process().memory_info().rss / 1e6


def _build_layers():
    LAYERS_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    total = 0
    for i in range(NLAYERS):
        W = (rng.standard_normal((D, D)).astype(np.float32) * 0.02)
        np.save(LAYERS_DIR / f"L{i}.npy", W)
        total += W.nbytes
    return total


def run_load_all(x):
    """load EVERY layer into RAM, then forward. Peak RAM = whole model."""
    Ws = [np.load(LAYERS_DIR / f"L{i}.npy") for i in range(NLAYERS)]
    peak = _rss_mb()
    t = time.perf_counter()
    h = x
    for W in Ws:
        h = np.maximum(h @ W, 0)        # relu(matmul)
        peak = max(peak, _rss_mb())
    dt = time.perf_counter() - t
    del Ws; gc.collect()
    return h, peak, dt


def run_streamed(x):
    """load ONE layer at a time from disk, use it, free it. Peak RAM = one layer."""
    peak = _rss_mb()
    t = time.perf_counter()
    h = x
    for i in range(NLAYERS):
        W = np.load(LAYERS_DIR / f"L{i}.npy")   # only this layer enters RAM
        h = np.maximum(h @ W, 0)
        peak = max(peak, _rss_mb())
        del W                                    # free before next
    dt = time.perf_counter() - t
    return h, peak, dt


def main():
    total = _build_layers()
    x = np.random.default_rng(1).standard_normal((1, D)).astype(np.float32)
    base = _rss_mb()
    print(f"model: {NLAYERS} layers x {D}x{D} = {total/1e6:.0f} MB total on disk", flush=True)
    print(f"baseline process RSS: {base:.0f} MB\n", flush=True)

    # warm caches equally
    run_streamed(x); run_load_all(x)

    la_peak = st_peak = 0; la_t = st_t = 0.0
    for _ in range(REPS):
        gc.collect()
        _, p, dt = run_load_all(x); la_peak = max(la_peak, p); la_t += dt
        gc.collect()
        _, p, dt = run_streamed(x); st_peak = max(st_peak, p); st_t += dt
    la_t /= REPS; st_t /= REPS

    la_used = la_peak - base
    st_used = st_peak - base
    print(f"  LOAD-ALL  : peak +{la_used:6.0f} MB | {la_t*1e3:6.1f} ms", flush=True)
    print(f"  STREAMED  : peak +{st_used:6.0f} MB | {st_t*1e3:6.1f} ms", flush=True)
    print(f"\n  MEMORY: streamed uses {la_used/max(st_used,1):.0f}x LESS RAM "
          f"(~one layer {total/NLAYERS/1e6:.0f} MB vs whole model {total/1e6:.0f} MB)", flush=True)
    print(f"  SPEED COST: streamed is {st_t/la_t:.1f}x slower (disk reads per layer)", flush=True)
    print(f"\n  -> a model that does NOT fit in RAM can still run: peak RAM = one layer.", flush=True)
    print(f"     T2 (memory) and T3 (speed) trade off: less RAM costs read time.", flush=True)

    OUT.write_text(json.dumps({"D": D, "n_layers": NLAYERS, "total_mb": round(total/1e6, 1),
                   "load_all_peak_mb": round(la_used, 1), "streamed_peak_mb": round(st_used, 1),
                   "memory_reduction_x": round(la_used/max(st_used, 1), 1),
                   "load_all_ms": round(la_t*1e3, 1), "streamed_ms": round(st_t*1e3, 1),
                   "speed_cost_x": round(st_t/la_t, 2),
                   "note": "paged/streamed inference: peak RAM = one layer not whole model; "
                           "lets a model bigger than RAM run, at a disk-read speed cost."},
                   indent=2), encoding="utf-8")
    # cleanup
    for f in LAYERS_DIR.glob("*.npy"):
        f.unlink()
    print(f"\nwritten {OUT}", flush=True)


if __name__ == "__main__":
    main()
