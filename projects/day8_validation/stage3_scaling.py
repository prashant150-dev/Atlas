"""Day-8 Stage 3: does the VQ advantage SURVIVE SCALING? (critique #1, #18)

We cannot run a 400B model on 8 GB, so we measure the TREND of VQ's advantage over
scalar quantization as size grows — the evidence a reviewer wants for "why should
this survive scaling?". Two probes:

  A. Real GPT-2 matrices spanning 0.59M -> 38.6M weights (65x range): at ~2 bits,
     compare VQ vs scalar-ternary reconstruction NMSE; track advantage ratio vs size.
  B. Controlled scaling law: matrices of growing dimension n with FIXED intrinsic
     structure (low-rank r=n/4 + noise); measure VQ vs scalar advantage vs n.

Hypothesis: VQ exploits cross-weight correlation, present at least as much in larger
tensors, so the advantage holds or grows with size.

Honest scope: reconstruction-error scaling, not end-to-end capability scaling.

Run from repo root::

    python projects/day8_validation/stage3_scaling.py
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
sys.path.insert(0, str(_HERE.parent / "day6_vector_quant"))
from vq_vs_scalar import nmse, scalar_ternary, vector_quant  # type: ignore  # noqa: E402

OUT = _HERE / "stage3_results.json"
LOG = _HERE / "stage3_log.jsonl"
SEED = 0


def _log(r):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(r) + "\n"); fh.flush()


def _compare(W):
    rs, _ = scalar_ternary(W)
    rv, bpw = vector_quant(W, 4, 256, seed=SEED)
    return nmse(W, rs), nmse(W, rv), bpw


def probe_a():
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained("models/gpt2", local_files_only=True)
    sd = m.state_dict()
    names = [
        ("attn.c_proj (0.59M)", "transformer.h.0.attn.c_proj.weight"),
        ("attn.c_attn (1.77M)", "transformer.h.0.attn.c_attn.weight"),
        ("mlp.c_fc (2.36M)", "transformer.h.0.mlp.c_fc.weight"),
        ("wte embed (38.6M)", "transformer.wte.weight"),
    ]
    mats = [(lbl, sd[n].detach().cpu().float().numpy()) for lbl, n in names]
    del m
    rows = []
    print("Probe A - real GPT-2 matrices (size trend):", flush=True)
    for lbl, W in mats:
        t0 = time.perf_counter()
        e_s, e_v, bpw = _compare(W)
        ratio = e_s / e_v
        row = {"probe": "A", "matrix": lbl, "n_weights": int(W.size),
               "scalar_nmse": round(e_s, 5), "vq_nmse": round(e_v, 5),
               "vq_advantage_x": round(ratio, 3), "vq_bpw": round(bpw, 3),
               "sec": round(time.perf_counter() - t0, 1)}
        rows.append(row); _log(row)
        print(f"  {lbl:22s} | scalar {e_s:.4f} | vq {e_v:.4f} | VQ {ratio:.2f}x better", flush=True)
    return rows


def probe_b():
    rng = np.random.default_rng(SEED)
    rows = []
    print("Probe B - controlled scaling law (fixed structure, growing n):", flush=True)
    for n in (64, 128, 256, 512, 1024, 2048):
        out = 4 * n
        r = max(1, n // 4)
        A = rng.standard_normal((n, r)).astype(np.float32)
        B = rng.standard_normal((r, out)).astype(np.float32)
        W = (A @ B) / np.sqrt(r) + 0.3 * rng.standard_normal((n, out)).astype(np.float32)
        t0 = time.perf_counter()
        e_s, e_v, bpw = _compare(W)
        ratio = e_s / e_v
        row = {"probe": "B", "n": n, "n_weights": int(W.size),
               "scalar_nmse": round(e_s, 5), "vq_nmse": round(e_v, 5),
               "vq_advantage_x": round(ratio, 3), "sec": round(time.perf_counter() - t0, 1)}
        rows.append(row); _log(row)
        print(f"  n={n:5d} ({W.size/1e6:5.2f}M) | scalar {e_s:.4f} | vq {e_v:.4f} | VQ {ratio:.2f}x better", flush=True)
    return rows


def main():
    LOG.write_text("", encoding="utf-8")
    a = probe_a()
    b = probe_b()
    payload = {"note": "VQ advantage = scalar_nmse / vq_nmse; >1 = VQ better; rising/stable = survives scaling (reconstruction proxy)",
               "probe_a_real_gpt2": a, "probe_b_controlled": b}
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
