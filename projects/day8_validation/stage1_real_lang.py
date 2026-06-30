"""Day-8 Stage 1: does VQ+healing survive REAL English, with statistics + ablation?

Answers the reviewer critique head-on for the SIZE lever:
  * real language (GPT-2 = a real English LM) on held-out English prose, not the
    synthetic char_lm / keyed-substitution tasks,
  * 5 seeds -> mean ± std (not a single lucky seed),
  * full ABLATION at matched ~2 bits/weight: FP, scalar-ternary post-hoc,
    scalar-ternary + heal, VQ post-hoc, VQ + heal.

Honest scope: standard benchmarks (WikiText/TinyStories) need a download this
offline box lacks, so eval is a held-out English passage. This validates VQ+heal
on real language; it does NOT yet test MoE-on-language or downstream task accuracy
(those remain open, by design).

Run from repo root::

    python projects/day8_validation/stage1_real_lang.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "day6_vector_quant"))
sys.path.insert(0, str(_HERE.parent / "day4_healing_ceiling"))
from vq_vs_scalar import scalar_ternary, vector_quant  # type: ignore  # noqa: E402
from p3_vq_heal import VQConv1D, wrap_vq_student  # type: ignore  # noqa: E402
from corpus import TRAIN_TEXT  # type: ignore  # noqa: E402
from src.compression.healing_qat import (  # noqa: E402
    _distillation_loss, compute_bits_per_weight, wrap_ternary_student)

OUT = _HERE / "stage1_results.json"
LOG = _HERE / "stage1_log.jsonl"
_MODEL = "models/gpt2"
_WRAP = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
SEEDS = [0, 1, 2, 3, 4]
HEAL_STEPS = 40
SEQ_LEN = 64
D, K = 4, 256

# Held-out English prose (real natural language; disjoint from the healing corpus).
EVAL_TEXT = (
    "When the storm finally passed, the harbor lay quiet under a sky the colour of "
    "wet slate. Fishermen counted their losses and their luck in the same breath, "
    "knowing the sea gives and takes without explanation. A child asked her "
    "grandfather why the waves had been so angry, and he said only that the ocean "
    "keeps no grudges and remembers no kindness; it simply obeys the wind and the "
    "moon. In the market the next morning, prices rose and fell as rumours spread "
    "about the damage upriver. An old teacher reminded her students that history is "
    "rarely written by those who lived it, and that the truth of any event depends "
    "on who survives to tell it. Far to the north, engineers argued over a bridge "
    "that had stood for a century and now leaned a little more each winter. Some "
    "wanted to tear it down; others believed that patience and small repairs would "
    "outlast any grand rebuilding. The argument, like most arguments, was really "
    "about fear dressed up as opinion."
)


def _eval_ids(tok):
    ids = tok(EVAL_TEXT, return_tensors="pt").input_ids[0]
    if ids.numel() < SEQ_LEN:
        ids = ids.repeat((SEQ_LEN // ids.numel()) + 1)
    return ids[:SEQ_LEN].unsqueeze(0)


@torch.inference_mode()
def _ppl(model, ids):
    lg = model(ids).logits
    sl = lg[:, :-1, :].reshape(-1, lg.size(-1)).float()
    return float(torch.exp(F.cross_entropy(sl, ids[:, 1:].reshape(-1))).item())


def _apply_recon(model, method_fn):
    for name, p in model.named_parameters():
        if name.endswith(_WRAP):
            W = p.detach().cpu().float().numpy()
            recon, _ = method_fn(W)
            p.data.copy_(torch.from_numpy(recon.astype(np.float32)).reshape(p.shape))


def _heal(model, teacher, train_batches):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=5e-4)
    model.train()
    for b in train_batches:
        with torch.inference_mode():
            tl = teacher(b).logits
        tl = tl.clone()
        opt.zero_grad(set_to_none=True)
        loss = _distillation_loss(model(b).logits, tl, b, 2.0, 0.1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    model.eval()


def _train_batches(tok, seed):
    g = torch.Generator().manual_seed(seed)
    wins = []
    for t in TRAIN_TEXT:
        ids = tok(t, return_tensors="pt").input_ids[0]
        if ids.numel() < SEQ_LEN:
            ids = ids.repeat((SEQ_LEN // max(1, ids.numel())) + 1)
        wins.append(ids[:SEQ_LEN].unsqueeze(0))
    order = torch.randint(0, len(wins), (HEAL_STEPS,), generator=g)
    return [wins[i] for i in order]


def _fresh():
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True)


def _log(r):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(r) + "\n"); fh.flush()


def main():
    from transformers import AutoTokenizer
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    LOG.write_text("", encoding="utf-8")
    tok = AutoTokenizer.from_pretrained(_MODEL, local_files_only=True)
    ids = _eval_ids(tok)
    teacher = _fresh().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    fp_ppl = _ppl(teacher, ids)
    print(f"FP teacher ppl (real English) = {fp_ppl:.2f}", flush=True)
    _log({"method": "FP", "ppl": round(fp_ppl, 3), "seed": "-"})

    # ternary post-hoc is deterministic -> one run
    m = _fresh().eval(); _apply_recon(m, scalar_ternary)
    tern_ph = _ppl(m, ids); del m
    print(f"ternary post-hoc (det) ppl = {tern_ph:.1f}", flush=True)
    _log({"method": "ternary_posthoc", "ppl": round(tern_ph, 3), "seed": "det"})

    results = {"ternary_posthoc": [tern_ph]}

    def collect(method):
        results.setdefault(method, [])

    for s in SEEDS:
        # VQ post-hoc (seed-dependent k-means)
        m = _fresh().eval(); _apply_recon(m, lambda W: vector_quant(W, D, K, seed=s))
        v = _ppl(m, ids); del m
        results.setdefault("vq_posthoc", []).append(v)
        _log({"method": "vq_posthoc", "seed": s, "ppl": round(v, 3)})

        # ternary + heal
        m = _fresh(); wrap_ternary_student(m, train_shadow=True)
        _heal(m, teacher, _train_batches(tok, s))
        th = _ppl(m, ids); del m
        results.setdefault("ternary_heal", []).append(th)
        _log({"method": "ternary_heal", "seed": s, "ppl": round(th, 3)})

        # VQ + heal
        m = _fresh(); wrap_vq_student(m, D, K, s)
        _heal(m, teacher, _train_batches(tok, s))
        vh = _ppl(m, ids); del m
        results.setdefault("vq_heal", []).append(vh)
        _log({"method": "vq_heal", "seed": s, "ppl": round(vh, 3)})

        print(f"seed {s} | vq_ph {v:8.1f} | tern_heal {th:8.1f} | vq_heal {vh:8.1f}", flush=True)

    def stat(xs):
        return {"mean": round(statistics.mean(xs), 2),
                "std": round(statistics.pstdev(xs), 2) if len(xs) > 1 else 0.0,
                "n": len(xs), "runs": [round(x, 2) for x in xs]}

    summary = {"FP": fp_ppl, "seeds": SEEDS, "heal_steps": HEAL_STEPS,
               "eval": "held-out English prose (offline)", "bits_per_weight": "~2.0",
               "ablation": {k: stat(v) for k, v in results.items()}}
    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print("\n=== ABLATION (mean ± std ppl, real English, 5 seeds) ===", flush=True)
    print(f"  FP teacher            : {fp_ppl:.1f}", flush=True)
    for k in ("ternary_posthoc", "vq_posthoc", "ternary_heal", "vq_heal"):
        st = stat(results[k])
        print(f"  {k:18s} : {st['mean']:8.1f} ± {st['std']:.1f}", flush=True)
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
