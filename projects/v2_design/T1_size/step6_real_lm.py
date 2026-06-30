"""T1 Step 6 — does RigL native-sparse beat post-hoc on a REAL LM (GPT-2)?

The toy showed RigL (gradient regrowth) keeps 83% of dense at 98% sparse where post-hoc
dies. Now validate on REAL GPT-2 weights with healing (distillation from the FP teacher),
measuring held-out PERPLEXITY at high sparsity. Weights stay FP here to isolate the
SPARSITY effect (quantization is a separate, already-proven lever).

Two arms per sparsity:
  * post-hoc : prune smallest-|w| to s, FIXED mask, then heal
  * RigL     : start at s, EVOLVE the mask (drop smallest-|w| active + grow largest-|grad|
               inactive, from dense grads) during healing

Run from repo root::

    python projects/v2_design/T1_size/step6_real_lm.py [sparsity ...]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "day6_vector_quant"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "day4_healing_ceiling"))
from corpus import TRAIN_TEXT  # type: ignore
from p3_vq_heal import SEED, SEQ_LEN, _batches, _eval_ids, _ppl, _WRAP  # type: ignore
from src.compression.healing_qat import _distillation_loss  # type: ignore

OUT = Path(__file__).resolve().parent / "step6_results.json"
_MODEL = "models/gpt2"
STEPS = 60
RIGL_EVERY = 15
LR = 5e-4


class SparseConv1D(nn.Module):
    """GPT-2 Conv1D weight with a sparsity mask (FP weight, trainable; mask evolvable)."""

    def __init__(self, weight, bias, sparsity, mode):
        super().__init__()
        W = weight.detach().clone().float()
        self.in_f, self.out_f = W.shape[0], W.shape[1]
        self.weight = nn.Parameter(W)
        self.register_buffer("mask", torch.ones_like(W))
        self.register_buffer("bias", None if bias is None else bias.detach().clone().float())
        self.sparsity = sparsity
        self._init_magnitude_mask(sparsity)

    def _init_magnitude_mask(self, s):
        f = self.weight.detach().abs().view(-1)
        k = int(len(f) * s)
        if k > 0:
            self.mask.copy_((self.weight.detach().abs() > f.kthvalue(k).values).float())

    @torch.no_grad()
    def rigl_step(self, drop_frac):
        """drop smallest-|w| active, grow largest-|grad| inactive (needs weight.grad set)."""
        if self.weight.grad is None:
            return
        w = self.weight.detach(); g = self.weight.grad.detach().abs()
        m = self.mask
        active = m.bool().view(-1); inactive = ~active
        n_active = int(active.sum()); n_inactive = int(inactive.sum())
        n_move = min(int(drop_frac * n_active), n_active, n_inactive)
        if n_move < 1:
            return
        wflat = w.abs().view(-1); gflat = g.view(-1); mflat = m.view(-1)
        ai = active.nonzero(as_tuple=True)[0]
        drop = ai[wflat[ai].topk(n_move, largest=False).indices]
        ii = inactive.nonzero(as_tuple=True)[0]
        grow = ii[gflat[ii].topk(n_move, largest=True).indices]
        mflat[drop] = 0.0; mflat[grow] = 1.0
        self.weight.data.view(-1)[grow] = 0.0

    def forward(self, x):
        w = self.weight * self.mask
        size_out = x.size()[:-1] + (self.out_f,)
        out = x.reshape(-1, x.size(-1)).float().matmul(w)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(size_out).to(dtype=x.dtype)


def wrap(model, sparsity, mode):
    by = dict(model.named_modules()); n = 0
    for name, _m in list(model.named_modules()):
        if name.endswith(_WRAP):
            parent = by[name.rpartition(".")[0]]; child = name.rpartition(".")[2]
            conv = getattr(parent, child)
            setattr(parent, child, SparseConv1D(conv.weight, getattr(conv, "bias", None),
                                                sparsity, mode))
            n += 1
    return n


def _sparse_modules(model):
    return [m for m in model.modules() if isinstance(m, SparseConv1D)]


def _heal(model, teacher, batches, eval_ids, mode, sparsity):
    import math
    mods = _sparse_modules(model)
    params = [m.weight for m in mods] + [p for n, p in model.named_parameters()
                                         if "weight" not in n or "ln" in n]
    params = list({id(p): p for p in model.parameters() if p.requires_grad}.values())
    opt = torch.optim.Adam(params, lr=LR)
    for step in range(STEPS):
        b = batches[step]
        with torch.inference_mode():
            tl = teacher(b).logits.clone()
        # RigL mask migration (dense grads) BEFORE the masked step
        if mode == "rigl" and step and step % RIGL_EVERY == 0 and step < STEPS * 0.8:
            saved = [m.mask.clone() for m in mods]   # remember the sparse mask
            for m in mods:                           # dense forward: temporarily unmask
                m.mask.fill_(1.0)
            opt.zero_grad(set_to_none=True)
            _distillation_loss(model(b).logits, tl, b, 2.0, 0.1).backward()
            for m, sv in zip(mods, saved):           # RESTORE sparse mask, keep dense grad
                m.mask.copy_(sv)
            df = 0.3 * 0.5 * (1 + math.cos(math.pi * step / (STEPS * 0.8)))
            for m in mods:
                m.rigl_step(df)                      # uses sparse mask + dense grad
            opt.zero_grad(set_to_none=True)
        # normal masked healing step
        opt.zero_grad(set_to_none=True)
        loss = _distillation_loss(model(b).logits, tl, b, 2.0, 0.1)
        loss.backward()
        for m in mods:                           # keep grads only on active weights
            if m.weight.grad is not None:
                m.weight.grad.mul_(m.mask)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    model.eval()
    return _ppl(model, eval_ids)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(SEED); torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    sparsities = [float(x) for x in sys.argv[1:]] or [0.90, 0.95]

    tok = AutoTokenizer.from_pretrained(_MODEL, local_files_only=True)
    teacher = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    eval_ids = _eval_ids(tok)
    teacher_ppl = _ppl(teacher, eval_ids)
    batches = _batches(tok, TRAIN_TEXT, SEQ_LEN, STEPS, SEED)
    print(f"FP teacher ppl {teacher_ppl:.2f}\n", flush=True)

    rows = []
    for s in sparsities:
        for mode in ("posthoc", "rigl"):
            m = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True)
            wrap(m, s, mode)
            ppl = _heal(m, teacher, batches, eval_ids, mode, s)
            rows.append({"sparsity": s, "mode": mode, "ppl": round(ppl, 2),
                         "vs_fp": round(ppl/teacher_ppl, 2)})
            print(f"  sparsity {s*100:3.0f}% | {mode:8s} | ppl {ppl:8.2f} | {ppl/teacher_ppl:.1f}x FP",
                  flush=True)
            del m

    print(f"\nFP {teacher_ppl:.1f}", flush=True)
    for s in sparsities:
        ph = next(r for r in rows if r["sparsity"] == s and r["mode"] == "posthoc")["ppl"]
        rg = next(r for r in rows if r["sparsity"] == s and r["mode"] == "rigl")["ppl"]
        win = "RigL WINS" if rg < ph else "post-hoc better"
        print(f"  {s*100:3.0f}% sparse: post-hoc {ph:.0f} vs RigL {rg:.0f} -> {win}", flush=True)
    payload = {"model": _MODEL, "teacher_ppl": round(teacher_ppl, 2), "rows": rows,
               "note": "Step 6: RigL native-sparse vs post-hoc on REAL GPT-2 (FP weights + "
                       "healing), held-out perplexity. Validates the toy finding on a real LM."}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
