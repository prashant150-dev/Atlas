"""Day-16 AetherFold probe: is there CROSS-LAYER redundancy to exploit? (new tech)

All prior compression (ours + SOTA) is per-matrix. The new idea: transformer layers
resemble each other, so coding each layer's DELTA from a prediction (previous layer
/ running base) could cost far fewer bits than coding raw weights — a floor BELOW
the per-matrix entropy D1 measured.

This probe decides FAST whether the idea has signal, before building the full method:
  1. cross-layer cosine similarity of same-role matrices across GPT-2's 12 blocks,
  2. entropy(raw weight) vs entropy(delta from previous layer) — if delta entropy is
     lower, delta-coding wins,
  3. NMSE of VQ on raw vs VQ on deltas at the same codebook (does delta compress better).

If deltas are lower-entropy / better-compressed -> green light to build AetherFold.

Run:  python projects/day16_aetherfold/probe_crosslayer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "day6_vector_quant"))
from vq_vs_scalar import _assign, _kmeans, nmse  # type: ignore  # noqa: E402

OUT = _HERE / "probe_results.json"


def gaussian_entropy_bits(x):
    v = float(np.var(x)) + 1e-12
    return 0.5 * np.log2(2 * np.pi * np.e * v)


def vq_nmse(W, d=4, K=256, seed=0):
    a = W.reshape(-1)
    pad = (-a.size) % d
    if pad:
        a = np.concatenate([a, np.zeros(pad, a.dtype)])
    V = a.reshape(-1, d)
    c = _kmeans(V, K, seed=seed); idx = _assign(V, c)
    return nmse(W.reshape(-1), c[idx].reshape(-1)[:W.size])


def main():
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained("models/gpt2", local_files_only=True)
    sd = m.state_dict()
    n_layer = m.config.n_layer
    roles = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
    del m

    print(f"GPT-2: {n_layer} layers. Cross-layer redundancy probe.\n", flush=True)
    summary = {"n_layer": n_layer, "roles": {}}
    for role in roles:
        Ws = [sd[f"transformer.h.{i}.{role}"].detach().cpu().float().numpy() for i in range(n_layer)]
        # 1. adjacent-layer cosine similarity (flattened)
        cos = []
        for i in range(1, n_layer):
            a, b = Ws[i].reshape(-1), Ws[i-1].reshape(-1)
            cos.append(float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12)))
        # 2. entropy: raw vs delta-from-previous
        raw_H = np.mean([gaussian_entropy_bits(Ws[i]) for i in range(n_layer)])
        delta_H = np.mean([gaussian_entropy_bits(Ws[i] - Ws[i-1]) for i in range(1, n_layer)])
        # 3. VQ nmse: raw vs delta (same K) on a mid layer
        i = n_layer // 2
        raw_vq = vq_nmse(Ws[i])
        delta_vq = vq_nmse(Ws[i] - Ws[i-1])
        summary["roles"][role] = {
            "adjacent_cosine_mean": round(float(np.mean(cos)), 4),
            "raw_entropy_bits": round(float(raw_H), 3),
            "delta_entropy_bits": round(float(delta_H), 3),
            "entropy_saving_bits": round(float(raw_H - delta_H), 3),
            "raw_vq_nmse": round(raw_vq, 4),
            "delta_vq_nmse": round(delta_vq, 4),
        }
        s = summary["roles"][role]
        print(f"{role:20s} | adj-cos {s['adjacent_cosine_mean']:+.3f} | "
              f"H raw {s['raw_entropy_bits']:.2f} -> delta {s['delta_entropy_bits']:.2f} "
              f"(save {s['entropy_saving_bits']:+.2f} b) | VQ nmse raw {s['raw_vq_nmse']:.3f} "
              f"delta {s['delta_vq_nmse']:.3f}", flush=True)

    # verdict
    avg_save = np.mean([v["entropy_saving_bits"] for v in summary["roles"].values()])
    avg_cos = np.mean([v["adjacent_cosine_mean"] for v in summary["roles"].values()])
    summary["avg_entropy_saving_bits"] = round(float(avg_save), 3)
    summary["avg_adjacent_cosine"] = round(float(avg_cos), 3)
    green = avg_save > 0.1 or avg_cos > 0.2
    summary["green_light"] = bool(green)
    print(f"\nAVG: adjacent-cosine {avg_cos:+.3f} | entropy saving {avg_save:+.3f} bits/weight", flush=True)
    print(f"VERDICT: {'GREEN — cross-layer structure exists, build AetherFold' if green else 'RED — layers too independent, delta-coding will not help'}", flush=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
