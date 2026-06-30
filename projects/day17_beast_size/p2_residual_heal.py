"""Part-1 BEAST SIZE — lever 2: residual / additive VQ + healing.

Mixed-precision (lever 1) only fixes the worst ~5% of vectors; the BULK error of the
other 95% is untouched (why p5->p10 stalled). Residual VQ attacks the bulk: quantize
the weight, then quantize the LEFTOVER error with a second codebook, so
    weight ~= C1[i] + C2[j]   (additive, AQLM-style).
At EQUAL bits this should beat a single codebook, because two coarse stages place
mass more flexibly than one. Then we stack lever-1 protection on the winner.

Equal-bits comparison (~2 bits/weight, codebooks heal from FP teacher, held-out ppl):
  * single  K=256          (log2 256 / 4              = 2.00 index bits)
  * residual M=2 x K=16    (2 * log2 16 / 4           = 2.00 index bits)
  * residual M=2 x K=16 + protect 5% @ int8  (lever 1 + lever 2 stacked)

Run from repo root::

    python projects/day17_beast_size/p2_residual_heal.py [arm ...]
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

OUT = _HERE / "p2_results.json"
_MODEL = "models/gpt2"
D = 4


class ResidualVQConv1D(nn.Module):
    """weight = sum_m codebook_m[idx_m]; M additive stages, each fit on the running
    residual. Optionally protect the worst `protect_frac` vectors at int8."""

    def __init__(self, weight, bias, d, K, M, protect_frac=0.0, seed=0):
        super().__init__()
        W = weight.detach().cpu().float().numpy()
        self.in_f, self.out_f = int(W.shape[0]), int(W.shape[1])
        self.d, self.K, self.M = d, K, M
        flat = W.reshape(-1)
        self.pad = int((-flat.size) % d)
        if self.pad:
            flat = np.concatenate([flat, np.zeros(self.pad, flat.dtype)])
        V = flat.reshape(-1, d).astype(np.float32)
        residual = V.copy()
        books, idxs = [], []
        for m in range(M):
            c = _kmeans(residual, K, seed=seed + m)
            i = _assign(residual, c)
            books.append(c)
            idxs.append(i)
            residual = residual - c[i]                      # peel off this stage
        # lever-1 protection on the FINAL residual's worst vectors
        err = (residual ** 2).sum(1)
        nprot = int(len(V) * protect_frac)
        self.nprot = nprot
        prot = np.argpartition(err, -nprot)[-nprot:] if nprot else np.empty(0, np.int64)
        for m in range(M):
            self.register_buffer(f"idx{m}", torch.from_numpy(idxs[m]).long())
            setattr(self, f"cb{m}", nn.Parameter(torch.from_numpy(books[m].astype(np.float32))))
        self.register_buffer("prot_idx", torch.from_numpy(prot.astype(np.int64)))
        self.prot = nn.Parameter(torch.from_numpy(V[prot].astype(np.float32))) if nprot else None
        self.register_buffer("bias", None if bias is None else bias.detach().clone().float())

    def weight(self):
        rec = getattr(self, "cb0")[getattr(self, "idx0")]
        for m in range(1, self.M):
            rec = rec + getattr(self, f"cb{m}")[getattr(self, f"idx{m}")]
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
        index_bits = self.M * math.log2(self.K) * nvec
        codebook_bits = self.M * self.K * self.d * 32
        protect_bits = self.nprot * self.d * 8 - self.nprot * self.M * math.log2(self.K)
        return (index_bits + codebook_bits + protect_bits) / (self.in_f * self.out_f)


def wrap(model, d, K, M, protect_frac, seed=0):
    by = dict(model.named_modules())
    n = 0
    for name, _m in list(model.named_modules()):
        if name.endswith(_WRAP):
            parent = by[name.rpartition(".")[0]]
            child = name.rpartition(".")[2]
            conv = getattr(parent, child)
            setattr(parent, child,
                    ResidualVQConv1D(conv.weight, getattr(conv, "bias", None), d, K, M, protect_frac, seed))
            n += 1
    return n


def _bpw(model):
    tw = sum(m.in_f * m.out_f for m in model.modules() if isinstance(m, ResidualVQConv1D))
    tb = sum(m.bits_per_weight() * m.in_f * m.out_f
             for m in model.modules() if isinstance(m, ResidualVQConv1D))
    return tb / tw


def _run_arm(arm, K, M, protect_frac, teacher, tb, eval_ids, t_argmax):
    from transformers import AutoModelForCausalLM
    st = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True)
    n = wrap(st, D, K, M, protect_frac, SEED)
    bpw = _bpw(st)
    print(f"{arm}: wrapped {n} | K={K} M={M} protect={protect_frac*100:.0f}% | "
          f"bits/weight {bpw:.3f}", flush=True)
    final = _heal(arm, st, teacher, tb, eval_ids, t_argmax, bpw)
    del st
    return {"arm": arm, "K": K, "M": M, "protect_frac": protect_frac,
            "bpw": round(bpw, 3), "ppl": final["ppl"], "top1": final["top1"]}


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

    specs = {
        "single_K256":     (256, 1, 0.0),    # baseline (= day-17 plain VQ)
        "resid_2xK16":     (16,  2, 0.0),    # equal ~2 bits, additive 2-stage
        "resid_2xK16_p5":  (16,  2, 0.05),   # + lever-1 protection (stacked)
    }
    want = sys.argv[1:] or list(specs)
    prior = json.loads(OUT.read_text(encoding="utf-8")).get("arms", []) if OUT.exists() else []
    arms = [a for a in prior if a["arm"] not in want]
    for name in want:
        K, M, pf = specs[name]
        arms.append(_run_arm(name, K, M, pf, teacher, tb, eval_ids, t_argmax))

    payload = {"model": _MODEL, "teacher_ppl": round(teacher_ppl, 2), "arms": arms,
               "note": "residual/additive VQ vs single codebook at equal bits, + stacked protection"}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nFP {teacher_ppl:.1f}", flush=True)
    for a in arms:
        print(f"  {a['arm']:16s} {a['bpw']:.2f} b/w | ppl {a['ppl']:8.2f} | "
              f"{a['ppl']/teacher_ppl:5.2f}x FP", flush=True)
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
