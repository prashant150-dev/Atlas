"""Part-1 BEAST SIZE — lever 1 on REAL quality: impact-weighted mixed-precision
VQ + healing, measured on perplexity (not NMSE).

The probe showed protecting the highest-error vectors at int8 lowers reconstruction
NMSE cheaply. But NMSE is not the goal — usable quality is. This experiment heals
three students against the FP teacher and compares HELD-OUT perplexity at honest
bits/weight:

  * VQ + heal            (p=0)  — our current best (plain 2-bit VQ, codebook healed)
  * Mixed-VQ + heal      (p=5%) — protect the worst 5% of weight-vectors at int8,
                                  the rest 2-bit VQ; codebook + protected both heal
  * VQ + heal, bigger K  control — spend the SAME extra bits on a larger codebook
                                  instead of protection (proves protection > just
                                  more codebook at equal size)

Win = Mixed-VQ closes the gap to FP meaningfully for its small extra bit cost AND
beats the equal-bits bigger-K control. Reuses the day-6 healing harness.

Run from repo root::

    python projects/day17_beast_size/p1_mixed_heal.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))            # repo root
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "day6_vector_quant"))             # vq + p3
sys.path.insert(0, str(_HERE.parent / "day4_healing_ceiling"))          # corpus
from vq_vs_scalar import _assign, _kmeans  # type: ignore  # noqa: E402
from corpus import TRAIN_TEXT  # type: ignore  # noqa: E402
from p3_vq_heal import (  # noqa: E402
    CHECKPOINTS, SEED, SEQ_LEN, _batches, _eval_ids, _heal, _ppl, _WRAP)

OUT = _HERE / "p1_results.json"
_MODEL = "models/gpt2"
D = 4


class MixedVQConv1D(nn.Module):
    """GPT-2 Conv1D: most d-vectors = trainable_codebook[fixed_idx]; the worst
    `protect_frac` (by k-means error) are kept as trainable int8-grade FP rows."""

    def __init__(self, weight, bias, d, K, protect_frac, seed=0):
        super().__init__()
        W = weight.detach().cpu().float().numpy()
        self.in_f, self.out_f = int(W.shape[0]), int(W.shape[1])
        self.d, self.K = d, K
        flat = W.reshape(-1)
        self.pad = int((-flat.size) % d)
        if self.pad:
            flat = np.concatenate([flat, np.zeros(self.pad, flat.dtype)])
        V = flat.reshape(-1, d)
        cent = _kmeans(V, K, seed=seed)
        idx = _assign(V, cent)
        err = ((V - cent[idx]) ** 2).sum(1)
        nprot = int(len(V) * protect_frac)
        self.nprot = nprot
        prot = np.argpartition(err, -nprot)[-nprot:] if nprot else np.empty(0, np.int64)
        self.register_buffer("idx", torch.from_numpy(idx).long())
        self.register_buffer("prot_idx", torch.from_numpy(prot.astype(np.int64)))
        self.codebook = nn.Parameter(torch.from_numpy(cent.astype(np.float32)))
        # protected rows: trainable FP (deployment int8 -> counted at 8 bits below)
        self.prot = nn.Parameter(torch.from_numpy(V[prot].astype(np.float32))) if nprot \
            else None
        self.register_buffer("bias", None if bias is None else bias.detach().clone().float())

    def weight(self):
        rec = self.codebook[self.idx]                      # (nvec, d)
        if self.nprot:
            rec = rec.clone()
            rec[self.prot_idx] = self.prot
        rec = rec.reshape(-1)[: self.in_f * self.out_f]
        return rec.reshape(self.in_f, self.out_f)

    def forward(self, x):
        w = self.weight()
        size_out = x.size()[:-1] + (self.out_f,)
        out = x.reshape(-1, x.size(-1)).float().matmul(w)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(size_out).to(dtype=x.dtype)

    def bits_per_weight(self):
        nvec = (self.in_f * self.out_f + self.pad) / self.d
        index_bits = math.log2(self.K) * nvec
        codebook_bits = self.K * self.d * 32
        protect_bits = self.nprot * self.d * 8 - self.nprot * math.log2(self.K)  # int8 rows replace their index
        return (index_bits + codebook_bits + protect_bits) / (self.in_f * self.out_f)


def wrap_mixed(model, d, K, protect_frac, seed=0):
    by = dict(model.named_modules())
    n = 0
    for name, _m in list(model.named_modules()):
        if name.endswith(_WRAP):
            parent = by[name.rpartition(".")[0]]
            child = name.rpartition(".")[2]
            conv = getattr(parent, child)
            setattr(parent, child,
                    MixedVQConv1D(conv.weight, getattr(conv, "bias", None), d, K, protect_frac, seed))
            n += 1
    return n


def _model_bpw(model):
    tot_w = sum(m.in_f * m.out_f for m in model.modules() if isinstance(m, MixedVQConv1D))
    tot_b = sum(m.bits_per_weight() * m.in_f * m.out_f
                for m in model.modules() if isinstance(m, MixedVQConv1D))
    return tot_b / tot_w


def _run_arm(arm, K, protect_frac, teacher, tb, eval_ids, t_argmax):
    from transformers import AutoModelForCausalLM
    st = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True)
    n = wrap_mixed(st, D, K, protect_frac, SEED)
    bpw = _model_bpw(st)
    print(f"{arm}: wrapped {n} layers | K={K} protect={protect_frac*100:.0f}% | "
          f"bits/weight {bpw:.3f}", flush=True)
    final = _heal(arm, st, teacher, tb, eval_ids, t_argmax, bpw)
    del st
    return {"arm": arm, "K": K, "protect_frac": protect_frac, "bpw": round(bpw, 3),
            "ppl": final["ppl"], "top1": final["top1"]}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    tok = AutoTokenizer.from_pretrained(_MODEL, local_files_only=True)
    teacher = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    eval_ids = _eval_ids(tok)
    with torch.inference_mode():
        t_argmax = teacher(eval_ids).logits[0].float().argmax(-1)
    teacher_ppl = _ppl(teacher, eval_ids)
    tb = _batches(tok, TRAIN_TEXT, SEQ_LEN, max(CHECKPOINTS), SEED)
    print(f"FP teacher ppl {teacher_ppl:.2f} (held-out passage)\n", flush=True)

    # arm specs; select via argv to stay inside time budget, results merged in OUT.
    specs = {
        "vq_K256_p0":    (256, 0.0),    # baseline plain 2-bit VQ
        "mixed_K256_p5": (256, 0.05),   # protect worst 5% at int8
        "vq_K512_p0":    (512, 0.0),    # equal-bits control: bigger codebook, no protect
        "mixed_K256_p2": (256, 0.02),   # lighter protection
        "mixed_K256_p10": (256, 0.10),  # heavier protection — does the gap keep closing?
    }
    want = sys.argv[1:] or list(specs)
    prior = json.loads(OUT.read_text(encoding="utf-8")).get("arms", []) if OUT.exists() else []
    arms = [a for a in prior if a["arm"] not in want]
    for name in want:
        K, pf = specs[name]
        arms.append(_run_arm(name, K, pf, teacher, tb, eval_ids, t_argmax))

    payload = {"model": _MODEL, "teacher_ppl": round(teacher_ppl, 2), "arms": arms,
               "note": "Part-1 beast-size: mixed-precision VQ+heal vs plain vs bigger-K, "
                       "held-out ppl. Win = mixed closest to FP and beats bigger-K control."}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nFP {teacher_ppl:.1f}", flush=True)
    for a in arms:
        gap = a["ppl"] / teacher_ppl
        print(f"  {a['arm']:16s} {a['bpw']:.2f} b/w | ppl {a['ppl']:8.2f} | {gap:5.1f}x FP",
              flush=True)
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
