"""Part-1 BEAST SIZE — final lever: SENSITIVITY-weighted protection (SqueezeLLM-style).

Lever-1 protects the d-vectors with the largest RECONSTRUCTION error. But the right
question is which weights most affect the LOSS, not which are hardest to reconstruct.
SqueezeLLM/GPTQ pick outliers by SENSITIVITY = (gradient of the loss w.r.t. weight)^2
(a diagonal-Hessian proxy). A weight with tiny reconstruction error can still be loss-
critical if the model is very sensitive to it.

This completes Part-1's "where do the bits go?" question: at EQUAL bits and EQUAL
healing, does protecting the most SENSITIVE 5% beat protecting the highest-ERROR 5%?

Selection score per d-vector = sum over its weights of  sensitivity * quant_error.
(SqueezeLLM weights the squared quant error by sensitivity; we protect the top 5%.)

Run from repo root::

    python projects/day17_beast_size/p4_sensitivity.py [arm ...]
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "day6_vector_quant"))
sys.path.insert(0, str(_HERE.parent / "day4_healing_ceiling"))
from vq_vs_scalar import _assign, _kmeans  # type: ignore  # noqa: E402
from corpus import TRAIN_TEXT  # type: ignore  # noqa: E402
from p3_vq_heal import (  # noqa: E402
    CHECKPOINTS, SEED, SEQ_LEN, _batches, _eval_ids, _heal, _ppl, _WRAP)

OUT = _HERE / "p4_results.json"
_MODEL = "models/gpt2"
D, K = 4, 256


def _sensitivity(model, tok):
    """Per-parameter (grad)^2 on a calibration batch = diagonal-Hessian proxy.
    Returns {param_name: tensor of squared grads}."""
    ids = tok(TRAIN_TEXT[0] + " " + TRAIN_TEXT[1], return_tensors="pt").input_ids[:, :SEQ_LEN]
    model.zero_grad(set_to_none=True)
    out = model(ids, labels=ids)
    out.loss.backward()
    sens = {}
    for name, p in model.named_parameters():
        if p.grad is not None and name.endswith(tuple(w + ".weight" for w in _WRAP)):
            sens[name] = (p.grad.detach().float() ** 2).cpu()
    model.zero_grad(set_to_none=True)
    return sens


class SensVQConv1D(nn.Module):
    """VQ; protect top `protect_frac` d-vectors chosen by `score` ('error' or 'sens')."""

    def __init__(self, weight, bias, sens, d, K, protect_frac, score, seed=0):
        super().__init__()
        W = weight.detach().cpu().float().numpy()
        self.in_f, self.out_f = int(W.shape[0]), int(W.shape[1])
        self.d, self.K = d, K
        flat = W.reshape(-1)
        self.pad = int((-flat.size) % d)
        sflat = sens.reshape(-1).numpy() if sens is not None else np.ones_like(flat)
        if self.pad:
            flat = np.concatenate([flat, np.zeros(self.pad, flat.dtype)])
            sflat = np.concatenate([sflat, np.zeros(self.pad, sflat.dtype)])
        V = flat.reshape(-1, d).astype(np.float32)
        S = sflat.reshape(-1, d).astype(np.float32)
        cent = _kmeans(V, K, seed=seed); idx = _assign(V, cent)
        sq_err = (V - cent[idx]) ** 2
        if score == "sens":
            vec_score = (S * sq_err).sum(1)            # sensitivity-weighted error
        else:
            vec_score = sq_err.sum(1)                  # plain reconstruction error
        nprot = int(len(V) * protect_frac)
        self.nprot = nprot
        prot = np.argpartition(vec_score, -nprot)[-nprot:] if nprot else np.empty(0, np.int64)
        self.register_buffer("idx", torch.from_numpy(idx).long())
        self.register_buffer("prot_idx", torch.from_numpy(prot.astype(np.int64)))
        self.codebook = nn.Parameter(torch.from_numpy(cent.astype(np.float32)))
        self.prot = nn.Parameter(torch.from_numpy(V[prot].astype(np.float32))) if nprot else None
        self.register_buffer("bias", None if bias is None else bias.detach().clone().float())

    def weight(self):
        rec = self.codebook[self.idx]
        if self.nprot:
            rec = rec.clone(); rec[self.prot_idx] = self.prot
        return rec.reshape(-1)[: self.in_f * self.out_f].reshape(self.in_f, self.out_f)

    def forward(self, x):
        w = self.weight()
        size_out = x.size()[:-1] + (self.out_f,)
        out = x.reshape(-1, x.size(-1)).float().matmul(w)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(size_out).to(dtype=x.dtype)

    def bits_per_weight(self):
        nvec = (self.in_f * self.out_f + self.pad) / self.d
        bits = math.log2(self.K) * nvec + self.K * self.d * 32 \
            + self.nprot * self.d * 8 - self.nprot * math.log2(self.K)
        return bits / (self.in_f * self.out_f)


def wrap(model, sens, d, K, protect_frac, score, seed=0):
    by = dict(model.named_modules()); n = 0
    for name, _m in list(model.named_modules()):
        if name.endswith(_WRAP):
            parent = by[name.rpartition(".")[0]]; child = name.rpartition(".")[2]
            conv = getattr(parent, child)
            s = sens.get(name + ".weight") if sens else None
            setattr(parent, child,
                    SensVQConv1D(conv.weight, getattr(conv, "bias", None), s, d, K, protect_frac, score, seed))
            n += 1
    return n


def _bpw(model):
    tw = sum(m.in_f * m.out_f for m in model.modules() if isinstance(m, SensVQConv1D))
    tb = sum(m.bits_per_weight() * m.in_f * m.out_f
             for m in model.modules() if isinstance(m, SensVQConv1D))
    return tb / tw


def _run(arm, score, teacher, sens, tb, eval_ids, t_argmax):
    from transformers import AutoModelForCausalLM
    st = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True)
    n = wrap(st, sens, D, K, 0.05, score, SEED)
    bpw = _bpw(st)
    print(f"{arm}: wrapped {n} | score={score} protect=5% | bits/weight {bpw:.3f}", flush=True)
    final = _heal(arm, st, teacher, tb, eval_ids, t_argmax, bpw)
    del st
    return {"arm": arm, "score": score, "bpw": round(bpw, 3),
            "ppl": final["ppl"], "top1": final["top1"]}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    tok = AutoTokenizer.from_pretrained(_MODEL, local_files_only=True)
    teacher = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True).eval()
    # sensitivity needs grads -> compute on a grad-enabled copy, then freeze teacher
    sens_model = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True)
    print("computing per-weight sensitivity (grad^2) on calibration batch...", flush=True)
    sens = _sensitivity(sens_model, tok); del sens_model
    for p in teacher.parameters():
        p.requires_grad_(False)
    eval_ids = _eval_ids(tok)
    with torch.inference_mode():
        t_argmax = teacher(eval_ids).logits[0].float().argmax(-1)
    teacher_ppl = _ppl(teacher, eval_ids)
    tb = _batches(tok, TRAIN_TEXT, SEQ_LEN, max(CHECKPOINTS), SEED)
    print(f"FP teacher ppl {teacher_ppl:.2f}\n", flush=True)

    specs = {"protect_error": "error", "protect_sens": "sens"}
    want = sys.argv[1:] or list(specs)
    prior = json.loads(OUT.read_text(encoding="utf-8")).get("arms", []) if OUT.exists() else []
    arms = [a for a in prior if a["arm"] not in want]
    for name in want:
        arms.append(_run(name, specs[name], teacher, sens, tb, eval_ids, t_argmax))

    payload = {"model": _MODEL, "teacher_ppl": round(teacher_ppl, 2), "arms": arms,
               "note": "equal-bits, equal-heal: sensitivity-weighted vs reconstruction-error "
                       "selection of the protected 5%. Win = lower ppl picks better weights."}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nFP {teacher_ppl:.1f}", flush=True)
    for a in arms:
        print(f"  {a['arm']:16s} {a['bpw']:.2f} b/w | ppl {a['ppl']:8.2f} | "
              f"{a['ppl']/teacher_ppl:5.2f}x FP", flush=True)
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
