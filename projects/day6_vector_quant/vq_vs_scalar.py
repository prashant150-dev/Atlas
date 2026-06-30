"""Day-6 (P-A): vector quantization vs scalar quantization on real GPT-2 weights.

Day-1 proved a ~2.04 bits/weight floor — but that is the *per-weight marginal*
entropy (cost of coding each weight independently). Coding GROUPS of weights
against a shared codebook (vector quantization) can go below it by exploiting
cross-weight correlation (joint entropy < sum of marginals). This is the
genuinely-new P-A lever (the mechanism behind AQLM / QuIP# / BTC-LLM).

This experiment tests the core claim HONESTLY on real GPT-2 weight matrices:
  at EQUAL bits/weight, does vector quantization beat scalar quantization on
  reconstruction error (NMSE)?

We sweep scalar (int b-bit, and ternary) and VQ (group size d, codebook size K),
compute honest bits/weight INCLUDING the codebook overhead, and compare NMSE.

Run from repo root::

    python projects/day6_vector_quant/vq_vs_scalar.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "vq_results.json"
LOG = _HERE / "vq_log.jsonl"
_MODEL = "models/gpt2"
SEED = 0


def nmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(((a - b) ** 2).sum() / (a ** 2).sum())


# ---- scalar baselines -----------------------------------------------------
def scalar_int(W: np.ndarray, bits: int) -> tuple[np.ndarray, float]:
    """Symmetric per-matrix int quantization. Returns (reconstruction, bits/weight)."""
    levels = (1 << (bits - 1)) - 1
    scale = np.abs(W).max() / max(levels, 1)
    q = np.clip(np.round(W / scale), -levels, levels)
    return q * scale, float(bits)


def scalar_ternary(W: np.ndarray, threshold_factor: float = 0.7) -> tuple[np.ndarray, float]:
    """Per-column ternary {-1,0,1}*scale. ~1.58 bits info, 2-bit packed + scales."""
    thr = threshold_factor * np.abs(W).mean(axis=0, keepdims=True)
    s = np.zeros_like(W)
    s[W > thr] = 1.0
    s[W < -thr] = -1.0
    kept = s != 0
    scale = (np.abs(W) * kept).sum(axis=0) / np.clip(kept.sum(axis=0), 1, None)
    recon = s * scale
    bpw = 2.0 + 32.0 * W.shape[1] / W.size      # 2-bit codes + fp32 per-column scale
    return recon, float(bpw)


# ---- vector quantization --------------------------------------------------
def _kmeans(X: np.ndarray, K: int, iters: int = 12, sample: int = 40000, seed: int = 0):
    """Tiny numpy k-means: fit centroids on a subsample, Lloyd iterations."""
    rng = np.random.default_rng(seed)
    fit = X if X.shape[0] <= sample else X[rng.choice(X.shape[0], sample, replace=False)]
    cent = fit[rng.choice(fit.shape[0], K, replace=False)].copy()
    for _ in range(iters):
        # assign fit points to nearest centroid (chunked)
        idx = _assign(fit, cent)
        for k in range(K):
            m = idx == k
            if m.any():
                cent[k] = fit[m].mean(axis=0)
    return cent


def _assign(X: np.ndarray, cent: np.ndarray, chunk: int = 20000) -> np.ndarray:
    out = np.empty(X.shape[0], dtype=np.int64)
    c2 = (cent ** 2).sum(axis=1)
    for i in range(0, X.shape[0], chunk):
        xb = X[i:i + chunk]
        d = (xb ** 2).sum(axis=1, keepdims=True) - 2 * xb @ cent.T + c2[None, :]
        out[i:i + chunk] = d.argmin(axis=1)
    return out


def vector_quant(W: np.ndarray, d: int, K: int, seed: int = 0) -> tuple[np.ndarray, float]:
    """Group rows of flattened W into d-dim vectors, k-means to K centroids."""
    flat = W.reshape(-1)
    pad = (-flat.size) % d
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, flat.dtype)])
    V = flat.reshape(-1, d)                       # [n_vec, d]
    cent = _kmeans(V, K, seed=seed)
    idx = _assign(V, cent)
    recon = cent[idx].reshape(-1)[: W.size].reshape(W.shape)
    # honest bits/weight: index bits per vector + codebook (fp32) amortised
    index_bits = np.log2(K) * V.shape[0]
    codebook_bits = K * d * 32
    bpw = (index_bits + codebook_bits) / W.size
    return recon, float(bpw)


def _log(row):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n"); fh.flush()


def main():
    from transformers import AutoModelForCausalLM
    np.random.seed(SEED)
    LOG.write_text("", encoding="utf-8")

    model = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True)
    sd = model.state_dict()
    # a representative GPT-2 weight matrix: block-0 MLP up-projection (Conv1D [768,3072])
    name = "transformer.h.0.mlp.c_fc.weight"
    W = sd[name].detach().cpu().float().numpy()
    print(f"matrix {name} shape {W.shape} ({W.size:,} weights)", flush=True)

    rows = []

    def record(method, recon, bpw):
        e = nmse(W, recon)
        row = {"method": method, "bits_per_weight": round(bpw, 4), "nmse": round(e, 6)}
        rows.append(row); _log(row)
        print(f"  {method:22s} | {bpw:6.3f} bits/wt | NMSE {e:.5f}", flush=True)
        return row

    print("scalar baselines:", flush=True)
    for b in (1, 2, 3, 4):
        r, bpw = scalar_int(W, b); record(f"scalar_int{b}", r, bpw)
    r, bpw = scalar_ternary(W); record("scalar_ternary", r, bpw)

    print("vector quantization:", flush=True)
    t0 = time.perf_counter()
    for d, K in [(4, 16), (8, 256), (4, 256), (2, 16), (4, 4096), (8, 4096)]:
        r, bpw = vector_quant(W, d, K, seed=SEED)
        record(f"vq_d{d}_K{K}", r, bpw)
    print(f"  (vq sweep {time.perf_counter()-t0:.1f}s)", flush=True)

    payload = {"matrix": name, "shape": list(W.shape), "n_weights": int(W.size),
               "results": rows,
               "note": "NMSE vs bits/weight; VQ beats scalar at equal bits => cross-weight structure is real"}
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
