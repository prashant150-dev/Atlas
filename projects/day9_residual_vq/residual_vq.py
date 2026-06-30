"""Day-9 A1: residual / additive VQ — push P-A below the 2-bit plain-VQ floor.

Plain VQ codes each weight-group with ONE codebook. Residual (additive) VQ codes
it as a SUM of entries from several codebooks: v ~= C1[i1] + C2[i2] + ... Each
extra codebook refines the residual, capturing structure a single codebook misses
(this is the core of AQLM). The question: at MATCHED bits/weight, does additive-of-
small beat single-large? And can it reach ~1.5 bits at usable quality where plain
VQ's sub-2-bit configs failed (Day-6 P4)?

We test reconstruction NMSE on a real GPT-2 matrix at matched bits/weight.

Run from repo root::

    python projects/day9_residual_vq/residual_vq.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "day6_vector_quant"))
from vq_vs_scalar import _assign, _kmeans, nmse, scalar_ternary, vector_quant  # type: ignore  # noqa: E402

OUT = _HERE / "a1_results.json"
LOG = _HERE / "a1_log.jsonl"
SEED = 0


def _vecs(W, d):
    a = W.reshape(-1)
    pad = (-a.size) % d
    if pad:
        a = np.concatenate([a, np.zeros(pad, a.dtype)])
    return a.reshape(-1, d), pad


def residual_vq(W, d, Ks, seed=0):
    """Additive VQ: v ~= sum_m Cm[im]. Returns (reconstruction, bits/weight)."""
    V, pad = _vecs(W, d)
    nvec = V.shape[0]
    recon = np.zeros_like(V)
    residual = V.copy()
    index_bits = 0.0
    codebook_bits = 0.0
    for K in Ks:
        c = _kmeans(residual, K, seed=seed)
        idx = _assign(residual, c)
        recon += c[idx]
        residual = residual - c[idx]
        index_bits += math.log2(K) * nvec
        codebook_bits += K * d * 32
    bpw = (index_bits + codebook_bits) / W.size
    return recon.reshape(-1)[: W.size].reshape(W.shape), float(bpw)


def _log(r):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(r) + "\n"); fh.flush()


def main():
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained("models/gpt2", local_files_only=True)
    W = m.state_dict()["transformer.h.0.mlp.c_fc.weight"].detach().cpu().float().numpy()
    del m
    print(f"matrix mlp.c_fc {W.shape} ({W.size:,} weights)\n", flush=True)

    rows = []

    def rec(label, recon, bpw):
        e = nmse(W, recon)
        row = {"method": label, "bits_per_weight": round(bpw, 3), "nmse": round(e, 5)}
        rows.append(row); _log(row)
        print(f"  {label:26s} | {bpw:5.3f} b/w | NMSE {e:.4f}", flush=True)

    print("baselines:", flush=True)
    r, b = scalar_ternary(W); rec("scalar_ternary", r, b)
    r, b = vector_quant(W, 4, 256, seed=SEED); rec("single_VQ d4K256", r, b)

    print("\nresidual/additive VQ (matched ~2 bits):", flush=True)
    t0 = time.perf_counter()
    r, b = residual_vq(W, 4, [16, 16], seed=SEED); rec("residual d4 [16,16]", r, b)
    r, b = residual_vq(W, 8, [256, 256], seed=SEED); rec("residual d8 [256,256]", r, b)

    print("\nsub-2-bit push (where plain VQ failed):", flush=True)
    r, b = residual_vq(W, 4, [16, 4], seed=SEED); rec("residual d4 [16,4] (~1.5b)", r, b)
    r, b = residual_vq(W, 8, [256, 16], seed=SEED); rec("residual d8 [256,16] (~1.25b)", r, b)
    r, b = vector_quant(W, 8, 256, seed=SEED); rec("single_VQ d8K256 (~1b, ref)", r, b)

    print("\nhigher-quality (3 codebooks):", flush=True)
    r, b = residual_vq(W, 4, [256, 64, 16], seed=SEED); rec("residual d4 [256,64,16]", r, b)
    print(f"\n(elapsed {time.perf_counter()-t0:.1f}s)", flush=True)

    payload = {"matrix": "transformer.h.0.mlp.c_fc.weight", "n_weights": int(W.size),
               "results": rows,
               "note": "residual/additive VQ vs single VQ at matched bits; lower NMSE at equal bits = win"}
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
