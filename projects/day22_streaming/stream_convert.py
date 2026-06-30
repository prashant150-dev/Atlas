"""Streaming converter: turn a model into AetherCore's 2-bit architecture AS IT ARRIVES,
tensor-by-tensor, so the full FP model is NEVER held in RAM or on disk.

This is how SOTA low-bit quant (GPTQ/AQLM) actually works and answers the user's idea:
"as a 400B model downloads, convert each chunk on the fly, discard the FP." We use
safetensors LAZY loading (`safe_open` -> `get_tensor(key)` loads one tensor on demand),
quantize that tensor with mixed-precision VQ (Part-1), write the compressed result, free
the FP tensor, move on. Peak RAM = the single largest tensor, NOT the whole model.

What this DOES solve: never needing the full FP model resident (800GB for a 400B model).
What it does NOT solve: the compressed output still needs disk (400B@2bit ~100GB), and
full behavioural healing is heavier (here we do fast per-tensor VQ; local-heal is a TODO
hook). Both stated honestly.

Run:  python projects/day22_streaming/stream_convert.py models/gpt2 experiments/gpt2_streamed
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "day6_vector_quant"))
from vq_vs_scalar import _assign, _kmeans  # type: ignore

DG, K = 4, 256
PROTECT = 0.05
# tensors we keep RAW (fp16): embeddings + norms + biases — 2-bit on these is garbage
# (the "jongjong" lesson). Quantize only the big 2D projection weights.
_RAW_SUBSTR = ("wte", "wpe", "ln", "bias", "norm")


def _mixed_vq(W):
    """mixed-precision VQ of a 2D weight -> compressed dict (indices+codebook+protected)."""
    shp = W.shape
    flat = W.reshape(-1).astype(np.float32)
    pad = int((-flat.size) % DG)
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, np.float32)])
    V = flat.reshape(-1, DG)
    cent = _kmeans(V, K, seed=0)
    idx = _assign(V, cent).astype(np.uint8)               # K<=256 -> 1 byte/group
    err = ((V - cent[idx]) ** 2).sum(1)
    nprot = int(len(V) * PROTECT)
    prot_pos = np.argpartition(err, -nprot)[-nprot:].astype(np.int32) if nprot else np.empty(0, np.int32)
    # protected rows stored at int8 (near-lossless), 4x cheaper than fp32
    prot_v = V[prot_pos] if nprot else np.empty((0, DG), np.float32)
    scale = (np.abs(prot_v).max(1, keepdims=True) / 127 + 1e-12) if nprot else np.ones((0, 1), np.float32)
    prot_q = np.round(prot_v / scale).clip(-127, 127).astype(np.int8) if nprot else np.empty((0, DG), np.int8)
    return {
        "kind": "mixed_vq", "shape": list(shp), "pad": pad,
        "idx": idx, "codebook": cent.astype(np.float16),
        "prot_pos": prot_pos, "prot_q": prot_q, "prot_scale": scale.astype(np.float16).reshape(-1),
    }


def _compressed_bytes(c):
    """honest stored size INCLUDING the protection positions + scales overhead."""
    if c["kind"] != "mixed_vq":
        return c["raw"].nbytes
    return (c["idx"].nbytes + c["codebook"].nbytes + c["prot_pos"].nbytes
            + c["prot_q"].nbytes + c["prot_scale"].nbytes)


def _dequant(c):
    """reconstruct fp32 weight (for round-trip verification)."""
    if c["kind"] != "mixed_vq":
        return c["raw"].astype(np.float32)
    cent = c["codebook"].astype(np.float32)
    rec = cent[c["idx"]].copy()
    if len(c["prot_pos"]):
        rec[c["prot_pos"]] = c["prot_q"].astype(np.float32) * c["prot_scale"][:, None].astype(np.float32)
    n = int(np.prod(c["shape"]))
    return rec.reshape(-1)[:n].reshape(c["shape"])


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "models/gpt2")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "experiments/gpt2_streamed")
    dst.mkdir(parents=True, exist_ok=True)
    from safetensors import safe_open

    sfile = src / "model.safetensors"
    full_fp_bytes = sfile.stat().st_size
    print(f"source: {sfile} ({full_fp_bytes/1e6:.0f} MB FP on disk)", flush=True)
    print("streaming tensor-by-tensor (lazy load) — full FP never resident:\n", flush=True)

    manifest = {"format": "aethercore-streamed-v1", "d": DG, "K": K, "protect": PROTECT,
                "tensors": {}}
    peak_fp_tensor = 0
    out_bytes = 0
    t0 = time.perf_counter()
    max_abs_err = 0.0
    n_q = n_raw = 0

    with safe_open(str(sfile), framework="numpy") as f:
        keys = list(f.keys())
        for i, key in enumerate(keys):
            W = f.get_tensor(key)                          # <-- ONLY this tensor enters RAM
            peak_fp_tensor = max(peak_fp_tensor, W.nbytes)
            quantize = (W.ndim == 2 and not any(s in key for s in _RAW_SUBSTR)
                        and min(W.shape) >= DG)
            if quantize:
                c = _mixed_vq(W.astype(np.float32))
                err = float(np.max(np.abs(_dequant(c) - W.astype(np.float32))))
                max_abs_err = max(max_abs_err, err)
                n_q += 1
            else:
                c = {"kind": "raw", "shape": list(W.shape), "raw": W.astype(np.float16)}
                n_raw += 1
            # write this tensor's compressed arrays NOW, then drop the FP tensor
            np.savez(dst / f"t{i:04d}.npz", **{k: v for k, v in c.items()
                                               if isinstance(v, np.ndarray)})
            meta = {k: v for k, v in c.items() if not isinstance(v, np.ndarray)}
            meta["file"] = f"t{i:04d}.npz"
            manifest["tensors"][key] = meta
            out_bytes += _compressed_bytes(c)
            del W, c                                       # free FP + compressed for this tensor
            if i % 20 == 0 or i == len(keys) - 1:
                print(f"  [{i+1:3d}/{len(keys)}] {key[:42]:42s} "
                      f"peak-FP-resident {peak_fp_tensor/1e6:5.1f} MB", flush=True)

    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    dt = time.perf_counter() - t0

    print(f"\n--- done in {dt:.1f}s ---", flush=True)
    print(f"  tensors: {n_q} quantized (2-bit mixed) + {n_raw} kept raw (embeds/norms)", flush=True)
    print(f"  FULL FP model size      : {full_fp_bytes/1e6:7.0f} MB", flush=True)
    print(f"  PEAK FP held in RAM     : {peak_fp_tensor/1e6:7.1f} MB  "
          f"({full_fp_bytes/peak_fp_tensor:.0f}x less than full model)", flush=True)
    print(f"  compressed output       : {out_bytes/1e6:7.1f} MB  "
          f"({full_fp_bytes/out_bytes:.1f}x smaller)", flush=True)
    print(f"  round-trip max |err|    : {max_abs_err:.4f} (quantized tensors)", flush=True)
    print(f"\n  KEY POINT: a {full_fp_bytes/1e6:.0f} MB model converted while never holding "
          f"more than {peak_fp_tensor/1e6:.0f} MB of FP at once.", flush=True)
    print(f"  For a 400B model the same loop holds only ~one layer (~hundreds of MB), not 800GB.",
          flush=True)

    res = {"full_fp_mb": round(full_fp_bytes/1e6, 1), "peak_fp_resident_mb": round(peak_fp_tensor/1e6, 1),
           "resident_reduction_x": round(full_fp_bytes/peak_fp_tensor, 1),
           "compressed_mb": round(out_bytes/1e6, 1), "compression_x": round(full_fp_bytes/out_bytes, 2),
           "n_quantized": n_q, "n_raw": n_raw, "max_abs_err": round(max_abs_err, 5),
           "seconds": round(dt, 1),
           "note": "streaming tensor-by-tensor conversion; peak RAM = largest single tensor, "
                   "not the whole model. Proves full FP need never be resident."}
    (Path(__file__).resolve().parent / "stream_results.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwritten manifest + {n_q+n_raw} compressed tensors to {dst}", flush=True)


if __name__ == "__main__":
    main()
