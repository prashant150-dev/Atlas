"""T2 MEMORY — deep dive toward 101x less RAM (paging x low-bit x prefetch).

101x memory math:
  memory_reduction  ~=  n_layers_paged  x  (16 / bits_per_weight)
  e.g. 80 layers paged + 2-bit weights = 80 x 8 = 640x  ->  101x is comfortably reachable.
The binding constraint is NOT RAM, it's the disk-READ time per layer. This deep dive
measures peak RAM and speed for 4 ways of running the SAME stacked-layer model:

  load_all (fp32)        : whole model in RAM (baseline; fast, huge RAM)
  paged (fp32)           : one fp32 layer at a time (low RAM, slow reads)
  paged (2-bit packed)   : one 2-bit layer at a time (lowest RAM, 16x smaller -> faster reads)
  paged 2-bit + PREFETCH : background thread reads next layer while current computes

No training; pure runtime engineering; CPU-friendly.

Run:  python projects/v2_design/T2_memory/deepdive_101x.py
"""

from __future__ import annotations

import gc
import json
import threading
import time
from pathlib import Path
from queue import Queue

import numpy as np
import psutil

HERE = Path(__file__).resolve().parent
LD = HERE / "_dd_layers"
OUT = HERE / "deepdive_results.json"

D = 4096
NLAYERS = 24
REPS = 3


def _rss():
    return psutil.Process().memory_info().rss / 1e6


def _pack_2bit(W):
    """ternary-quantize W and pack 4 weights/byte. Returns (codes_uint8, scale)."""
    scale = np.abs(W).mean() + 1e-9
    q = np.clip(np.round(W / scale), -1, 1).astype(np.int8)   # {-1,0,1}
    codes = (q + 1).astype(np.uint8)                          # {0,1,2}
    flat = codes.reshape(-1)
    pad = (-flat.size) % 4
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, np.uint8)])
    packed = (flat[0::4] | (flat[1::4] << 2) | (flat[2::4] << 4) | (flat[3::4] << 6))
    return packed.astype(np.uint8), np.float32(scale)


def _unpack_2bit(packed, scale, shape):
    n = shape[0] * shape[1]
    codes = np.empty(packed.size * 4, np.uint8)
    codes[0::4] = packed & 3
    codes[1::4] = (packed >> 2) & 3
    codes[2::4] = (packed >> 4) & 3
    codes[3::4] = (packed >> 6) & 3
    q = codes[:n].astype(np.float32) - 1.0
    return (q * scale).reshape(shape)


def _build():
    LD.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    fp32_bytes = packed_bytes = 0
    for i in range(NLAYERS):
        W = (rng.standard_normal((D, D)).astype(np.float32) * 0.02)
        np.save(LD / f"f{i}.npy", W)
        packed, scale = _pack_2bit(W)
        np.savez(LD / f"q{i}.npz", packed=packed, scale=scale)
        fp32_bytes += W.nbytes
        packed_bytes += packed.nbytes
    return fp32_bytes, packed_bytes


def run_load_all(x):
    Ws = [np.load(LD / f"f{i}.npy") for i in range(NLAYERS)]
    peak = _rss(); t = time.perf_counter(); h = x
    for W in Ws:
        h = np.maximum(h @ W, 0); peak = max(peak, _rss())
    dt = time.perf_counter() - t
    del Ws; gc.collect()
    return peak, dt


def run_paged_fp32(x):
    peak = _rss(); t = time.perf_counter(); h = x
    for i in range(NLAYERS):
        W = np.load(LD / f"f{i}.npy")
        h = np.maximum(h @ W, 0); peak = max(peak, _rss()); del W
    return peak, time.perf_counter() - t


def run_paged_2bit(x):
    peak = _rss(); t = time.perf_counter(); h = x
    for i in range(NLAYERS):
        z = np.load(LD / f"q{i}.npz")
        W = _unpack_2bit(z["packed"], z["scale"], (D, D))
        h = np.maximum(h @ W, 0); peak = max(peak, _rss()); del W, z
    return peak, time.perf_counter() - t


def run_paged_2bit_prefetch(x):
    """background thread reads+unpacks the next layer while we compute the current."""
    q = Queue(maxsize=2)

    def loader():
        for i in range(NLAYERS):
            z = np.load(LD / f"q{i}.npz")
            q.put(_unpack_2bit(z["packed"], z["scale"], (D, D)))
        q.put(None)

    th = threading.Thread(target=loader, daemon=True); th.start()
    peak = _rss(); t = time.perf_counter(); h = x
    while True:
        W = q.get()
        if W is None:
            break
        h = np.maximum(h @ W, 0); peak = max(peak, _rss()); del W
    th.join()
    return peak, time.perf_counter() - t


def main():
    fp32_b, packed_b = _build()
    x = np.random.default_rng(1).standard_normal((1, D)).astype(np.float32)
    base = _rss()
    print(f"model: {NLAYERS}x{D}x{D} | fp32 {fp32_b/1e6:.0f} MB | 2-bit packed {packed_b/1e6:.0f} MB "
          f"({fp32_b/packed_b:.0f}x smaller on disk)", flush=True)
    print(f"baseline RSS {base:.0f} MB\n", flush=True)

    # warm
    run_paged_fp32(x); run_paged_2bit(x); run_load_all(x); run_paged_2bit_prefetch(x)

    modes = {"load_all_fp32": run_load_all, "paged_fp32": run_paged_fp32,
             "paged_2bit": run_paged_2bit, "paged_2bit_prefetch": run_paged_2bit_prefetch}
    res = {}
    for name, fn in modes.items():
        pk = 0.0; tt = 0.0
        for _ in range(REPS):
            gc.collect(); p, dt = fn(x); pk = max(pk, p); tt += dt
        res[name] = {"peak_mb": round(pk - base, 1), "ms": round(tt/REPS*1e3, 1)}

    la = res["load_all_fp32"]["peak_mb"]
    print(f"{'mode':22s} {'peak RAM':>10} {'reduction':>10} {'time':>9}", flush=True)
    print("-" * 56, flush=True)
    for name, r in res.items():
        red = la / max(r["peak_mb"], 1)
        print(f"{name:22s} {r['peak_mb']:8.0f}MB {red:8.0f}x {r['ms']:7.0f}ms", flush=True)

    best = res["paged_2bit_prefetch"]
    print(f"\n  BEST (paged 2-bit + prefetch): {la/max(best['peak_mb'],1):.0f}x less RAM than "
          f"load-all, at {best['ms']/res['load_all_fp32']['ms']:.1f}x the time", flush=True)
    # projection: a deep real model
    print(f"\n  101x MEMORY projection: reduction ~= n_layers x (16/bits).", flush=True)
    print(f"    80-layer model, 2-bit, paged -> 80 x 8 = 640x less RAM (101x easily cleared).", flush=True)
    print(f"    A 70B fp16 model (140GB) -> ~one 2-bit layer (~0.2GB) resident.", flush=True)

    OUT.write_text(json.dumps({"D": D, "n_layers": NLAYERS, "fp32_mb": round(fp32_b/1e6, 1),
                   "packed_mb": round(packed_b/1e6, 1), "modes": res,
                   "note": "paging x low-bit x prefetch toward 101x memory; reduction=n_layers x "
                           "(16/bits); read time is the real cost, cut by low-bit + prefetch."},
                   indent=2), encoding="utf-8")
    for f in LD.glob("*"):
        f.unlink()
    print(f"\nwritten {OUT}", flush=True)


if __name__ == "__main__":
    main()
