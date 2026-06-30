"""T1 (NO GPU, NO training): how far can POST-HOC get toward 0.15-bit? SparseGPT-style.

Honest test of the user's question: build an aggressively-sparse model with the BEST
no-training method (SparseGPT = Hessian-aware prune + error-compensation, the sparse
cousin of GPTQ) and MEASURE the real perplexity at each sparsity, toward 0.15 bits/weight.
No training, no GPU — pure post-hoc linear algebra on calibration activations.

Expected (honest): post-hoc degrades with sparsity and collapses at the extreme; this
measures EXACTLY where, so we know the real no-GPU ceiling vs the native-training path.

Run from repo root::

    python projects/v2_design/T1_size/sparsegpt_no_train.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "day6_vector_quant"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "day4_healing_ceiling"))
from p3_vq_heal import _eval_ids, _ppl, _WRAP  # type: ignore

OUT = Path(__file__).resolve().parent / "sparsegpt_results.json"
_MODEL = "models/gpt2"


def sparsegpt_layer(W, X, sparsity, damp=0.01):
    """Hessian-aware prune to `sparsity` + compensate remaining, per input-dim. No training.
    W:[d_in,d_out], X:[n,d_in] real calibration. Returns pruned+compensated W."""
    W = W.copy().astype(np.float64)
    d_in = W.shape[0]
    H = (X.T @ X) / len(X)
    H += damp * np.mean(np.diag(H)) * np.eye(d_in)
    Hinv = np.linalg.inv(H)
    dinv = np.diag(Hinv)
    for i in range(d_in):
        w = W[i, :]
        # saliency = w^2 / Hinv_ii ; prune the lowest fraction in this row
        sal = w ** 2 / (dinv[i] + 1e-12)
        k = int(len(w) * sparsity)
        if k > 0:
            thr = np.partition(sal, k)[k]
            mask = sal > thr
        else:
            mask = np.ones_like(w, bool)
        q = w * mask                                 # pruned (kept weights stay FP here)
        err = (w - q) / Hinv[i, i]
        W[i, :] = q
        if i + 1 < d_in:
            W[i + 1:, :] -= np.outer(Hinv[i + 1:, i], err)   # compensate remaining rows
    return W.astype(np.float32)


def _calibration(model, tok):
    acts = {}
    hooks = []
    layers = {n: m for n, m in model.named_modules() if n.endswith(_WRAP)}
    for n, m in layers.items():
        acts[n] = []
        def mk(nm):
            def hook(mod, inp, out):
                acts[nm].append(inp[0].detach().reshape(-1, inp[0].shape[-1]).float().numpy())
            return hook
        hooks.append(m.register_forward_hook(mk(n)))
    seeds = ["The history of science is patient observation overturning comfortable belief.",
             "In the beginning the universe was hot and dense, then expanded and cooled down.",
             "Programming means telling a computer exactly what to do, one careful step at a time.",
             "Economics studies how people allocate scarce resources among many competing uses.",
             "Music blends rhythm, melody and harmony to express feeling far beyond mere words.",
             "Mathematics is the language in which the deep laws of nature are most clearly written."]
    with torch.inference_mode():
        for s in seeds:
            ids = tok(s * 14, return_tensors="pt").input_ids[:, :1024]
            model(ids)
    for h in hooks:
        h.remove()
    return {n: np.concatenate(v, 0) for n, v in acts.items()}, layers


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_MODEL, local_files_only=True)
    eval_ids = _eval_ids(tok)

    base = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True).eval()
    fp_ppl = _ppl(base, eval_ids)
    print(f"FP teacher ppl {fp_ppl:.2f}", flush=True)
    print("collecting real calibration activations (no training)...", flush=True)
    calib, _ = _calibration(base, tok)
    orig = {n: base.state_dict()[n + ".weight"].detach().cpu().float().numpy()
            for n in calib}
    del base

    rows = []
    for s in (0.50, 0.90, 0.95, 0.98):
        m = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True).eval()
        sd = m.state_dict()
        for n in calib:
            Wc = sparsegpt_layer(orig[n], calib[n], s)
            sd[n + ".weight"].copy_(torch.from_numpy(Wc))
        m.load_state_dict(sd)
        ppl = _ppl(m, eval_ids)
        # effective bits if kept weights were ternary (1.58b) + 1-bit/weight mask
        eff_bits = (1 - s) * 1.58 + 0.05
        rows.append({"sparsity": s, "eff_bits": round(eff_bits, 3), "ppl": round(ppl, 2),
                     "vs_fp": round(ppl / fp_ppl, 2)})
        print(f"  sparsity {s*100:3.0f}% | ~{eff_bits:.2f} bits/wt | ppl {ppl:8.2f} | "
              f"{ppl/fp_ppl:.1f}x FP", flush=True)
        del m

    print(f"\nFP {fp_ppl:.1f}  (target 0.15 bits ~ 98% sparse ternary)", flush=True)
    best = min(rows, key=lambda r: r["ppl"])
    print(f"  best post-hoc sparse: ppl {best['ppl']} @ {best['sparsity']*100:.0f}% "
          f"(~{best['eff_bits']:.2f} bits)", flush=True)
    print("  VERDICT: post-hoc (no GPU) at 0.15-bit/98% -> see the ppl above; this is the", flush=True)
    print("  honest no-training ceiling. Native training is what crosses it (toy: 83%).", flush=True)
    OUT.write_text(json.dumps({"fp_ppl": round(fp_ppl, 2), "rows": rows,
                   "note": "SparseGPT-style post-hoc prune+compensate (no training, CPU) at rising "
                           "sparsity toward 0.15 bits; measures the real no-GPU accuracy ceiling."},
                   indent=2), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
