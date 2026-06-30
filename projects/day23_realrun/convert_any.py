"""Generic streaming converter: ANY HF safetensors model -> AetherCore 2-bit format.

Generalises day22's GPT-2 converter to any architecture (Qwen/Llama/etc.):
  - loads tensors LAZILY via safetensors (framework='pt' so bf16/fp16/fp32 all work),
    across one OR many shard files, one tensor at a time -> bounded RAM.
  - quantizes the big 2D projection weights with mixed-precision VQ (Part-1);
  - keeps embeddings / norms / lm_head / biases raw (fp16) — 2-bit on those is garbage.

Run:  python projects/day23_realrun/convert_any.py models/qwen2.5-1.5b experiments/qwen_streamed
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "day6_vector_quant"))
from vq_vs_scalar import _assign, _kmeans  # type: ignore

DG, K = 4, 256
PROTECT = 0.05
# raw (not quantized): embeddings, all norms, lm head, biases. Substring match, lowercase.
_RAW_SUBSTR = ("embed", "norm", "lm_head", "bias", "wte", "wpe", "ln_")


def _mixed_vq(W):
    shp = W.shape
    flat = W.reshape(-1).astype(np.float32)
    pad = int((-flat.size) % DG)
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, np.float32)])
    V = flat.reshape(-1, DG)
    cent = _kmeans(V, K, seed=0)
    idx = _assign(V, cent).astype(np.uint8)
    err = ((V - cent[idx]) ** 2).sum(1)
    nprot = int(len(V) * PROTECT)
    pos = np.argpartition(err, -nprot)[-nprot:].astype(np.int32) if nprot else np.empty(0, np.int32)
    pv = V[pos] if nprot else np.empty((0, DG), np.float32)
    sc = (np.abs(pv).max(1, keepdims=True) / 127 + 1e-12) if nprot else np.ones((0, 1), np.float32)
    pq = np.round(pv / sc).clip(-127, 127).astype(np.int8) if nprot else np.empty((0, DG), np.int8)
    return {"kind": "mixed_vq", "shape": list(shp), "pad": pad, "idx": idx,
            "codebook": cent.astype(np.float16), "prot_pos": pos, "prot_q": pq,
            "prot_scale": sc.astype(np.float16).reshape(-1)}


def _csize(c):
    if c["kind"] != "mixed_vq":
        return c["raw"].nbytes
    return sum(c[k].nbytes for k in ("idx", "codebook", "prot_pos", "prot_q", "prot_scale"))


def main():
    src = Path(sys.argv[1]); dst = Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
    from safetensors import safe_open
    shards = sorted(src.glob("*.safetensors"))
    full_fp = sum(s.stat().st_size for s in shards)
    print(f"source: {src} | {len(shards)} shard(s) | {full_fp/1e9:.2f} GB FP", flush=True)

    manifest = {"format": "aethercore-streamed-v1", "d": DG, "K": K, "protect": PROTECT,
                "src": str(src), "tensors": {}}
    peak = out_bytes = 0
    n_q = n_raw = 0
    t0 = time.perf_counter()
    ti = 0
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                W = f.get_tensor(key)                      # one tensor in RAM (bf16/fp16/fp32)
                peak = max(peak, W.numel() * 4)            # fp32-equiv footprint
                Wn = W.float().numpy()
                low = key.lower()
                quant = (W.ndim == 2 and not any(s in low for s in _RAW_SUBSTR)
                         and min(W.shape) >= DG)
                if quant:
                    c = _mixed_vq(Wn); n_q += 1
                else:
                    c = {"kind": "raw", "shape": list(W.shape), "raw": Wn.astype(np.float16)}
                    n_raw += 1
                np.savez(dst / f"t{ti:05d}.npz",
                         **{k: v for k, v in c.items() if isinstance(v, np.ndarray)})
                meta = {k: v for k, v in c.items() if not isinstance(v, np.ndarray)}
                meta["file"] = f"t{ti:05d}.npz"
                manifest["tensors"][key] = meta
                out_bytes += _csize(c)
                del W, Wn, c
                if ti % 50 == 0:
                    print(f"  [{ti:4d}] {key[:46]:46s} peak {peak/1e6:6.1f} MB", flush=True)
                ti += 1
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    dt = time.perf_counter() - t0
    print(f"\n--- done {dt:.0f}s | {ti} tensors ({n_q} quantized, {n_raw} raw) ---", flush=True)
    print(f"  full FP            : {full_fp/1e9:6.2f} GB", flush=True)
    print(f"  PEAK resident      : {peak/1e6:6.0f} MB ({full_fp/peak:.0f}x less)", flush=True)
    print(f"  compressed output  : {out_bytes/1e9:6.2f} GB ({full_fp/out_bytes:.1f}x smaller)", flush=True)
    res = {"src": str(src), "full_fp_gb": round(full_fp/1e9, 3), "peak_mb": round(peak/1e6, 1),
           "compressed_gb": round(out_bytes/1e9, 3), "compression_x": round(full_fp/out_bytes, 2),
           "n_quantized": n_q, "n_raw": n_raw, "seconds": round(dt, 1)}
    (Path(__file__).resolve().parent / "convert_results.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print(f"written -> {dst}", flush=True)


if __name__ == "__main__":
    main()
