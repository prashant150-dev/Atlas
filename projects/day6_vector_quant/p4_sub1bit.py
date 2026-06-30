"""Day-6 P4: push VQ below 1 bit/weight (large groups) + healing.

P3 made VQ+heal the best ~2-bit lever. P4 chases sub-1-bit. For VQ the way down
is LARGER groups with small codebooks: bits/weight = log2(K)/d + codebook
overhead. We sweep several sub-1-bit configs post-hoc, then HEAL the most
promising one and see how close it gets to FP.

Honest note on "learnable transform": QuIP/QuaRot random rotations help *scalar*
quantization (they spread outliers across coordinates). For VQ a single per-group
rotation is k-means-invariant, and a random projection destroys the very
cross-weight structure VQ exploits — so it does NOT help here. We include a quick
rotation sanity-check, and rely on the VQ-appropriate levers (group size +
codebook healing).

Run from repo root::

    python projects/day6_vector_quant/p4_sub1bit.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(_HERE))
from vq_vs_scalar import _assign, _kmeans, nmse, vector_quant  # type: ignore  # noqa: E402
from p3_vq_heal import (  # noqa: E402
    VQConv1D, _batches, _eval_ids, _heal, _ppl, _top1, wrap_vq_student, TRAIN_TEXT)

OUT = _HERE / "p4_results.json"
LOG = _HERE / "p4_log.jsonl"
_MODEL = "models/gpt2"
SEED = 0
# (d, K) -> bits/weight ~ log2(K)/d (plus small codebook overhead)
SUB1_CONFIGS = [(4, 16), (6, 16), (8, 64), (8, 16), (16, 256)]


def vq_bpw(model):
    tot_w = 0
    tot_bits = 0.0
    for m in model.modules():
        if isinstance(m, VQConv1D):
            K, d = m.codebook.shape
            n = m.in_f * m.out_f
            tot_w += n
            tot_bits += math.log2(K) * ((n + m.pad) / d) + K * d * 32
    return tot_bits / max(tot_w, 1)


def _log(r):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(r) + "\n"); fh.flush()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    LOG.write_text("", encoding="utf-8")

    tok = AutoTokenizer.from_pretrained(_MODEL, local_files_only=True)
    teacher = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    eval_ids = _eval_ids(tok)
    with torch.inference_mode():
        t_argmax = teacher(eval_ids).logits[0].float().argmax(-1)
    teacher_ppl = _ppl(teacher, eval_ids)
    print(f"FP teacher ppl {teacher_ppl:.2f} (held-out)", flush=True)
    _log({"phase": "fp", "ppl": round(teacher_ppl, 2)})

    # --- rotation sanity check on one matrix: does it change VQ NMSE? ---
    W = teacher.state_dict()["transformer.h.0.mlp.c_fc.weight"].numpy().astype(np.float32)
    _, bpw0 = vector_quant(W, 8, 64, seed=SEED)
    r1, _ = vector_quant(W, 8, 64, seed=SEED)
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((8, 8)))      # per-group 8x8 rotation
    flat = W.reshape(-1)[: (W.size // 8) * 8].reshape(-1, 8)
    rot = (flat @ Q)
    cent = _kmeans(rot, 64, seed=SEED); idx = _assign(rot, cent)
    rot_recon = (cent[idx] @ Q.T)                         # rotate back
    nmse_plain = nmse(flat, r1.reshape(-1)[:flat.size].reshape(-1, 8))
    nmse_rot = nmse(flat, rot_recon)
    print(f"rotation check (d8K64): plain NMSE {nmse_plain:.4f} vs rotated {nmse_rot:.4f} "
          f"(~equal => rotation doesn't help VQ)", flush=True)
    _log({"phase": "rotation_check", "nmse_plain": round(nmse_plain, 5), "nmse_rotated": round(nmse_rot, 5)})

    # --- sub-1-bit post-hoc sweep ---
    print("sub-1-bit post-hoc sweep:", flush=True)
    rows = []
    for d, K in SUB1_CONFIGS:
        t0 = time.perf_counter()
        m = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True).eval()
        wrap_vq_student(m, d, K, SEED)
        bpw = vq_bpw(m)
        ppl = _ppl(m, eval_ids); top1 = _top1(m, t_argmax, eval_ids)
        row = {"phase": "posthoc", "d": d, "K": K, "bits_per_weight": round(bpw, 4),
               "ppl": round(ppl, 2), "top1": round(top1, 4), "sec": round(time.perf_counter() - t0, 1)}
        rows.append(row); _log(row)
        print(f"  d{d} K{K} | {bpw:5.3f} b/w | ppl {ppl:10.1f} | top1 {top1:.3f}", flush=True)
        del m

    # --- heal the best sub-1-bit config (lowest post-hoc ppl with bpw<1) ---
    sub1 = [r for r in rows if r["bits_per_weight"] < 1.0]
    best = min(sub1 or rows, key=lambda r: r["ppl"])
    print(f"healing best sub-1-bit: d{best['d']} K{best['K']} ({best['bits_per_weight']} b/w)", flush=True)
    sv = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True)
    wrap_vq_student(sv, best["d"], best["K"], SEED)
    bpw = vq_bpw(sv)
    train_batches = _batches(tok, TRAIN_TEXT, 64, 60, SEED)
    healed = _heal(f"vq_sub1_d{best['d']}K{best['K']}", sv, teacher, train_batches, eval_ids, t_argmax, bpw)

    payload = {"model": _MODEL, "teacher_ppl": teacher_ppl, "posthoc_sweep": rows,
               "healed_best": healed,
               "note": "sub-1-bit via large-group VQ + healing; rotation does not help VQ"}
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nFP {teacher_ppl:.1f} | best sub-1bit healed ({bpw:.3f} b/w) ppl {healed['ppl']}", flush=True)
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
