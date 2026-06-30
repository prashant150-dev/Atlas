"""Day-6 P3: VQ + healing vs ternary + healing — close the gap to FP.

P2: post-hoc VQ beats post-hoc ternary 26x on perplexity at equal bits, but both
are far from FP. P3 heals both and compares at EQUAL bits/weight (the real win
test):

  * ternary + healing : trainable FP shadow weights, STE (the D2 method).
  * VQ + healing       : FIX the k-means assignments, make the CODEBOOK trainable,
    distil the FP teacher into it. Bits/weight unchanged (indices fixed); only the
    256x4 codebook per matrix moves -> tiny number of trainable params.

Train on a held-out-disjoint corpus, eval perplexity/top-1 on a separate passage
(P1.1's lesson: never eval on the training text).

Run from repo root::

    python projects/day6_vector_quant/p3_vq_heal.py
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
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root (src.*)
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                                 # vq_vs_scalar
sys.path.insert(0, str(_HERE.parent / "day4_healing_ceiling")) # corpus (TRAIN_TEXT)
from vq_vs_scalar import _assign, _kmeans  # type: ignore  # noqa: E402
from corpus import TRAIN_TEXT  # type: ignore  # noqa: E402
from src.compression.healing_qat import (  # noqa: E402
    TernaryShadowConv1D, _distillation_loss, compute_bits_per_weight, wrap_ternary_student)

OUT = _HERE / "p3_results.json"
LOG = _HERE / "p3_log.jsonl"
_MODEL = "models/gpt2"
_WRAP = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
D, K = 4, 256
SEQ_LEN = 64
CHECKPOINTS = (0, 15, 30, 60)
LR = 5e-4
SEED = 0

_EVAL_TEXT = (
    "The history of science is a history of patient observation slowly overturning "
    "comfortable belief. A theory survives only when it keeps making predictions that "
    "could have failed but did not. The people who changed the world were rarely the "
    "loudest; they looked again at what everyone assumed was settled."
)


class VQConv1D(nn.Module):
    """GPT-2 Conv1D with weight = trainable_codebook[fixed_assignments]."""

    def __init__(self, weight, bias, d, K, seed=0):
        super().__init__()
        W = weight.detach().cpu().float().numpy()
        self.in_f, self.out_f = int(W.shape[0]), int(W.shape[1])
        flat = W.reshape(-1)
        self.pad = int((-flat.size) % d)
        if self.pad:
            flat = np.concatenate([flat, np.zeros(self.pad, flat.dtype)])
        V = flat.reshape(-1, d)
        cent = _kmeans(V, K, seed=seed)
        idx = _assign(V, cent)
        self.d = d
        self.register_buffer("idx", torch.from_numpy(idx).long())
        self.codebook = nn.Parameter(torch.from_numpy(cent.astype(np.float32)))
        self.register_buffer("bias", None if bias is None else bias.detach().clone().float())

    def weight(self):
        rec = self.codebook[self.idx].reshape(-1)[: self.in_f * self.out_f]
        return rec.reshape(self.in_f, self.out_f)

    def forward(self, x):
        w = self.weight()
        size_out = x.size()[:-1] + (self.out_f,)
        out = x.reshape(-1, x.size(-1)).float().matmul(w)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(size_out).to(dtype=x.dtype)


def wrap_vq_student(model, d, K, seed=0):
    n = 0
    by_name = dict(model.named_modules())
    for name, _m in list(model.named_modules()):
        if name.endswith(_WRAP):
            parent = by_name[name.rpartition(".")[0]]
            child = name.rpartition(".")[2]
            conv = getattr(parent, child)
            setattr(parent, child, VQConv1D(conv.weight, getattr(conv, "bias", None), d, K, seed))
            n += 1
    return n


def _batches(tokenizer, texts, seq_len, steps, seed):
    wins = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt").input_ids[0]
        if ids.numel() < seq_len:
            ids = ids.repeat((seq_len // max(1, ids.numel())) + 1)
        wins.append(ids[:seq_len].unsqueeze(0))
    return [wins[i % len(wins)] for i in range(steps)]


def _eval_ids(tokenizer):
    ids = tokenizer(_EVAL_TEXT, return_tensors="pt").input_ids[0]
    if ids.numel() < SEQ_LEN:
        ids = ids.repeat((SEQ_LEN // ids.numel()) + 1)
    return ids[:SEQ_LEN].unsqueeze(0)


@torch.inference_mode()
def _ppl(model, ids):
    lg = model(ids).logits
    sl = lg[:, :-1, :].reshape(-1, lg.size(-1)).float()
    return float(torch.exp(F.cross_entropy(sl, ids[:, 1:].reshape(-1))).item())


@torch.inference_mode()
def _top1(model, t_argmax, ids):
    return float((model(ids).logits[0].float().argmax(-1) == t_argmax).float().mean().item())


def _log(r):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(r) + "\n"); fh.flush()


def _heal(arm, student, teacher, train_batches, eval_ids, t_argmax, bpw):
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    done = 0
    for ck in CHECKPOINTS:
        student.train()
        for s in range(done, ck):
            b = train_batches[s]
            with torch.inference_mode():
                tl = teacher(b).logits
            tl = tl.clone()
            opt.zero_grad(set_to_none=True)
            loss = _distillation_loss(student(b).logits, tl, b, 2.0, 0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        done = ck
        student.eval()
        ppl = _ppl(student, eval_ids); top1 = _top1(student, t_argmax, eval_ids)
        row = {"arm": arm, "step": ck, "bits_per_weight": round(bpw, 3),
               "ppl": round(ppl, 2), "top1": round(top1, 4),
               "trainable_params": sum(p.numel() for p in params)}
        _log(row)
        print(f"  {arm:12s} step {ck:3d} | ppl {ppl:9.2f} | top1 {top1:.3f}", flush=True)
    return row


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
    train_batches = _batches(tok, TRAIN_TEXT, SEQ_LEN, max(CHECKPOINTS), SEED)
    print(f"FP teacher ppl {teacher_ppl:.2f} (held-out passage)", flush=True)
    _log({"arm": "FP_teacher", "step": -1, "bits_per_weight": 32, "ppl": round(teacher_ppl, 2), "top1": 1.0})

    # --- ternary + healing (D2 baseline) ---
    print("ternary + healing:", flush=True)
    st = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True)
    _, tern_params, tern_chan = wrap_ternary_student(st, train_shadow=True)
    tern_bpw = compute_bits_per_weight(tern_params, tern_chan)
    tern_final = _heal("ternary", st, teacher, train_batches, eval_ids, t_argmax, tern_bpw)
    del st

    # --- VQ + healing (assignments fixed, codebook trainable) ---
    print("VQ + healing:", flush=True)
    sv = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True)
    nv = wrap_vq_student(sv, D, K, SEED)
    # honest bits/weight for VQ: indices + codebook over wrapped weights
    tot_w = sum(m.in_f * m.out_f for m in sv.modules() if isinstance(m, VQConv1D))
    tot_bits = sum(math.log2(K) * ((m.in_f * m.out_f + m.pad) / D) + K * D * 32
                   for m in sv.modules() if isinstance(m, VQConv1D))
    vq_bpw = tot_bits / tot_w
    print(f"  wrapped {nv} layers, VQ bits/weight {vq_bpw:.3f}", flush=True)
    vq_final = _heal("vq", sv, teacher, train_batches, eval_ids, t_argmax, vq_bpw)

    payload = {"model": _MODEL, "d": D, "K": K, "teacher_ppl": teacher_ppl,
               "ternary_final": tern_final, "vq_final": vq_final,
               "note": "equal-bits VQ+heal vs ternary+heal, held-out eval; lower ppl = win"}
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nFP {teacher_ppl:.1f} | ternary+heal {tern_final['ppl']} | VQ+heal {vq_final['ppl']}", flush=True)
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
