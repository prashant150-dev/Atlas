"""T1 SIZE, Lever 2 — SPARSITY: how much can we prune on top of 2-bit VQ + heal?

Lever 1 (2-bit VQ) gave 8x. Lever 2 claims another ~4x from sparsity. But pruning costs
quality. This measures the REAL trade: zero out the smallest-magnitude weight-vectors at
increasing sparsity, keep the rest at 2-bit VQ, HEAL, and read held-out perplexity. The
honest multiplier is "how much sparsity survives healing".

Storage at sparsity s, 2-bit indices, with a 1-bit/vector mask:
   bits/weight ~= (1-s)*2  +  mask(1 bit / d weights)   -> falls toward 0 as s -> 1.

Run from repo root::

    python projects/v2_design/T1_size/lever2_sparsity.py [sparsity ...]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "day6_vector_quant"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "day4_healing_ceiling"))
from vq_vs_scalar import _assign, _kmeans  # type: ignore
from corpus import TRAIN_TEXT  # type: ignore
from p3_vq_heal import (CHECKPOINTS, SEED, SEQ_LEN, _batches, _eval_ids, _heal,  # type: ignore
                        _ppl, _WRAP)

OUT = _HERE / "lever2_results.json"
_MODEL = "models/gpt2"
D, K = 4, 256


class SparseVQConv1D(nn.Module):
    """2-bit VQ, but the smallest-magnitude `sparsity` fraction of d-vectors -> ZERO.
    Codebook is trainable (heals); the zero mask is fixed."""

    def __init__(self, weight, bias, sparsity, d, K, seed=0):
        super().__init__()
        W = weight.detach().cpu().float().numpy()
        self.in_f, self.out_f = int(W.shape[0]), int(W.shape[1])
        self.d, self.K, self.sparsity = d, K, sparsity
        flat = W.reshape(-1).astype(np.float32)
        self.pad = int((-flat.size) % d)
        if self.pad:
            flat = np.concatenate([flat, np.zeros(self.pad, np.float32)])
        V = flat.reshape(-1, d)
        # prune smallest-L2 vectors
        norms = (V ** 2).sum(1)
        nkill = int(len(V) * sparsity)
        kill = np.argpartition(norms, nkill)[:nkill] if nkill else np.empty(0, np.int64)
        keep_mask = np.ones(len(V), bool)
        keep_mask[kill] = False
        cent = _kmeans(V, K, seed=seed)
        idx = _assign(V, cent)
        self.register_buffer("idx", torch.from_numpy(idx).long())
        self.register_buffer("keep", torch.from_numpy(keep_mask))
        self.codebook = nn.Parameter(torch.from_numpy(cent.astype(np.float32)))
        self.register_buffer("bias", None if bias is None else bias.detach().clone().float())

    def weight(self):
        rec = self.codebook[self.idx]
        rec = rec * self.keep[:, None]                 # zero the pruned vectors
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
        kept = float(self.keep.float().mean())
        index_bits = math.log2(self.K) * nvec * kept     # only kept vectors need an index
        mask_bits = 1.0 * nvec                            # 1 bit/vector keep/drop mask
        codebook_bits = self.K * self.d * 32
        return (index_bits + mask_bits + codebook_bits) / (self.in_f * self.out_f)


def wrap(model, sparsity, seed=0):
    by = dict(model.named_modules()); n = 0
    for name, _m in list(model.named_modules()):
        if name.endswith(_WRAP):
            parent = by[name.rpartition(".")[0]]; child = name.rpartition(".")[2]
            conv = getattr(parent, child)
            setattr(parent, child, SparseVQConv1D(conv.weight, getattr(conv, "bias", None),
                                                  sparsity, D, K, seed))
            n += 1
    return n


def _bpw(model):
    tw = sum(m.in_f*m.out_f for m in model.modules() if isinstance(m, SparseVQConv1D))
    tb = sum(m.bits_per_weight()*m.in_f*m.out_f for m in model.modules() if isinstance(m, SparseVQConv1D))
    return tb / tw


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(SEED); torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    sparsities = [float(x) for x in sys.argv[1:]] or [0.0, 0.25, 0.50, 0.75]

    tok = AutoTokenizer.from_pretrained(_MODEL, local_files_only=True)
    teacher = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    eval_ids = _eval_ids(tok)
    with torch.inference_mode():
        t_argmax = teacher(eval_ids).logits[0].float().argmax(-1)
    teacher_ppl = _ppl(teacher, eval_ids)
    tb = _batches(tok, TRAIN_TEXT, SEQ_LEN, max(CHECKPOINTS), SEED)
    print(f"FP teacher ppl {teacher_ppl:.2f}\n", flush=True)

    rows = []
    for s in sparsities:
        st = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True)
        wrap(st, s, SEED)
        bpw = _bpw(st)
        size_x = 16.0 / bpw
        print(f"sparsity {s*100:3.0f}% | bits/weight {bpw:.3f} | size {size_x:.1f}x vs fp16", flush=True)
        final = _heal(f"sp{int(s*100)}", st, teacher, tb, eval_ids, t_argmax, bpw)
        rows.append({"sparsity": s, "bpw": round(bpw, 3), "size_x": round(size_x, 1),
                     "ppl": final["ppl"], "vs_fp": round(final["ppl"]/teacher_ppl, 2)})
        del st

    print(f"\nFP {teacher_ppl:.1f}", flush=True)
    for r in rows:
        print(f"  sparsity {r['sparsity']*100:3.0f}% | {r['bpw']:.2f} b/w | {r['size_x']:5.1f}x | "
              f"ppl {r['ppl']:7.2f} | {r['vs_fp']:.1f}x FP", flush=True)
    payload = {"model": _MODEL, "teacher_ppl": round(teacher_ppl, 2), "rows": rows,
               "note": "Lever 2 sparsity: prune smallest weight-vectors + 2-bit VQ + heal; "
                       "measures the real size-x vs quality trade as sparsity rises."}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
