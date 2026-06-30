"""Day-7: VQ + healing + MoE sparsity together — size AND effective-speed.

Two proven levers, now combined:
  * MoE sparsity: many experts, only top_k active per token -> huge TOTAL capacity
    but small ACTIVE compute/bandwidth.
  * VQ + healing (Day-6): store weights at ~2 bits with cross-weight codebooks.

Tension: VQ's codebook overhead kills tiny expert matrices. Fix: ALL experts of a
layer share ONE codebook (overhead amortised across every expert).

We train four variants on the char_lm task and report accuracy + honest bit
budgets (stored total, and FFN bits touched per token):
  1. DenseFP-small (H=128)   -- low capacity, low cost
  2. DenseFP-big   (H=1024)  -- high capacity, high cost (= MoE total capacity)
  3. MoE-FP        (8x128, top2) -- high total capacity, sparse active, FP stored
  4. VQ-MoE        (= MoE, experts at ~2-bit shared-codebook VQ + healing)

Thesis: VQ-MoE matches DenseFP-big accuracy at ~DenseFP-small stored bits and
~2-expert active bits — big-model capability, small stored+active cost.

Run from repo root::

    python projects/day7_vq_moe/vq_moe.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.architecture.experiment import TaskSpec, _evaluate, _make_char_lm_dataset, _train  # noqa: E402

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "results.json"
LOG = _HERE / "log.jsonl"

D_MODEL = 64
N_HEAD = 4
VOCAB = 29
SEQ = 24
N_EXPERT = 8
TOP_K = 2
EXP_H = 128
DG, K = 4, 256          # VQ group size, codebook size
SEED = 0
STEPS = 800
BATCH = 128
LR = 3e-3


# --------------------------------------------------------------------------
class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.nh = N_HEAD

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.nh, C // self.nh).transpose(1, 2)
        k = k.view(B, T, self.nh, C // self.nh).transpose(1, 2)
        v = v.view(B, T, self.nh, C // self.nh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // self.nh)
        mask = torch.triu(torch.ones(T, T), 1).bool()
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class DenseFFN(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.up = nn.Linear(D_MODEL, h)
        self.down = nn.Linear(h, D_MODEL)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))

    def ffn_bits(self):
        w = self.up.weight.numel() + self.down.weight.numel()
        return 32.0 * w, 32.0 * w     # stored, active (dense -> all used)


class MoEFFN(nn.Module):
    def __init__(self, n_expert, h, top_k):
        super().__init__()
        self.router = nn.Linear(D_MODEL, n_expert)
        self.up = nn.ModuleList([nn.Linear(D_MODEL, h) for _ in range(n_expert)])
        self.down = nn.ModuleList([nn.Linear(h, D_MODEL) for _ in range(n_expert)])
        self.n_expert, self.top_k, self.h = n_expert, top_k, h

    def forward(self, x):
        B, T, C = x.shape
        flat = x.reshape(-1, C)
        scores = self.router(flat)
        topv, topi = scores.topk(self.top_k, dim=-1)
        gate = topv.softmax(-1)
        out = torch.zeros_like(flat)
        for slot in range(self.top_k):
            idx = topi[:, slot]
            g = gate[:, slot:slot + 1]
            for e in range(self.n_expert):
                m = idx == e
                if m.any():
                    h = F.gelu(self.up[e](flat[m]))
                    out[m] += g[m] * self.down[e](h)
        return out.reshape(B, T, C)

    def ffn_bits(self):
        per = self.up[0].weight.numel() + self.down[0].weight.numel()
        stored = 32.0 * (per * self.n_expert + self.router.weight.numel())
        active = 32.0 * (per * self.top_k + self.router.weight.numel())
        return stored, active


def _vq_fit_shared(weights, d, K, seed):
    """Fit ONE codebook over the stacked vectors of many matrices.
    Returns (codebook[K,d], list of index arrays per matrix, list of shapes)."""
    from projects.day6_vector_quant.vq_vs_scalar import _assign, _kmeans  # local import
    vecs, shapes, sizes = [], [], []
    for W in weights:
        a = W.reshape(-1)
        pad = (-a.size) % d
        if pad:
            a = np.concatenate([a, np.zeros(pad, a.dtype)])
        v = a.reshape(-1, d)
        vecs.append(v); shapes.append(W.shape); sizes.append(v.shape[0])
    allv = np.concatenate(vecs, 0)
    cent = _kmeans(allv, K, seed=seed)
    idxs = []
    off = 0
    for s in sizes:
        idxs.append(_assign(allv[off:off + s], cent)); off += s
    return cent, idxs, shapes


class VQMoEFFN(nn.Module):
    """MoE whose expert weights are VQ-coded against per-layer shared codebooks."""

    def __init__(self, moe: MoEFFN, d, K, seed):
        super().__init__()
        self.router = moe.router
        self.n_expert, self.top_k, self.h, self.d = moe.n_expert, moe.top_k, moe.h, d
        up_w = [moe.up[e].weight.detach().cpu().numpy().astype(np.float32) for e in range(self.n_expert)]
        dn_w = [moe.down[e].weight.detach().cpu().numpy().astype(np.float32) for e in range(self.n_expert)]
        self.up_bias = nn.ParameterList([nn.Parameter(moe.up[e].bias.detach().clone()) for e in range(self.n_expert)])
        self.dn_bias = nn.ParameterList([nn.Parameter(moe.down[e].bias.detach().clone()) for e in range(self.n_expert)])
        up_cent, up_idx, self.up_shapes = _vq_fit_shared(up_w, d, K, seed)
        dn_cent, dn_idx, self.dn_shapes = _vq_fit_shared(dn_w, d, K, seed + 1)
        self.up_codebook = nn.Parameter(torch.from_numpy(up_cent.astype(np.float32)))
        self.dn_codebook = nn.Parameter(torch.from_numpy(dn_cent.astype(np.float32)))
        for e in range(self.n_expert):
            self.register_buffer(f"up_idx_{e}", torch.from_numpy(up_idx[e]).long())
            self.register_buffer(f"dn_idx_{e}", torch.from_numpy(dn_idx[e]).long())
        self.K = K

    def _w(self, codebook, idx, shape):
        n = shape[0] * shape[1]
        return codebook[idx].reshape(-1)[:n].reshape(shape)

    def forward(self, x):
        B, T, C = x.shape
        flat = x.reshape(-1, C)
        topv, topi = self.router(flat).topk(self.top_k, dim=-1)
        gate = topv.softmax(-1)
        out = torch.zeros_like(flat)
        for slot in range(self.top_k):
            idx = topi[:, slot]; g = gate[:, slot:slot + 1]
            for e in range(self.n_expert):
                m = idx == e
                if m.any():
                    uw = self._w(self.up_codebook, getattr(self, f"up_idx_{e}"), self.up_shapes[e])
                    dw = self._w(self.dn_codebook, getattr(self, f"dn_idx_{e}"), self.dn_shapes[e])
                    h = F.gelu(flat[m] @ uw.t() + self.up_bias[e])
                    out[m] += g[m] * (h @ dw.t() + self.dn_bias[e])
        return out.reshape(B, T, C)

    def ffn_bits(self):
        per = self.up_shapes[0][0] * self.up_shapes[0][1] + self.dn_shapes[0][0] * self.dn_shapes[0][1]
        index_bpw = math.log2(self.K) / self.d
        codebooks = 2 * self.K * self.d * 32
        stored = index_bpw * per * self.n_expert + codebooks + 32.0 * self.router.weight.numel()
        active = index_bpw * per * self.top_k + 32.0 * self.router.weight.numel()
        return stored, active


class LM(nn.Module):
    def __init__(self, ffn):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Parameter(torch.zeros(1, SEQ, D_MODEL))
        self.n1 = nn.LayerNorm(D_MODEL); self.n2 = nn.LayerNorm(D_MODEL)
        self.attn = Attn(); self.ffn = ffn
        self.head = nn.Linear(D_MODEL, VOCAB)

    def forward(self, tok):
        x = self.embed(tok) + self.pos[:, :tok.shape[1]]
        x = x + self.attn(self.n1(x))
        x = x + self.ffn(self.n2(x))
        return self.head(x)


def _log(r):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(r) + "\n"); fh.flush()


def main():
    torch.manual_seed(SEED)
    LOG.write_text("", encoding="utf-8")
    spec = TaskSpec(name="char_lm", vocab_size=VOCAB, seq_len=SEQ, chance_accuracy=0.168,
                    n_examples=4000, dense_steps=STEPS, aether_steps=STEPS,
                    dense_lr=LR, aether_lr=LR, batch_size=BATCH)
    gen = torch.Generator().manual_seed(0)
    X, Y = _make_char_lm_dataset(spec, gen)

    def run(label, model, steps=STEPS):
        f, l = _train(model, X, Y, steps, LR, BATCH, seed=0)
        acc, _ = _evaluate(model, X, Y)
        s, a = model.ffn.ffn_bits()
        row = {"variant": label, "acc": round(acc, 4), "ffn_stored_bits": int(s),
               "ffn_active_bits_per_token": int(a)}
        _log(row)
        print(f"  {label:16s} | acc {acc:.3f} | stored {s/1e3:7.1f}kb | active/tok {a/1e3:6.1f}kb", flush=True)
        return row, model

    t0 = time.perf_counter()
    rows = []
    print("training variants:", flush=True)
    r1, _ = run("DenseFP-small", LM(DenseFFN(EXP_H)))
    r2, _ = run("DenseFP-big", LM(DenseFFN(N_EXPERT * EXP_H)))
    r3, moe_model = run("MoE-FP", LM(MoEFFN(N_EXPERT, EXP_H, TOP_K)))
    rows += [r1, r2, r3]

    # VQ the trained MoE experts (shared codebook) + heal
    print("VQ-MoE (shared codebook) + healing:", flush=True)
    vq_ffn = VQMoEFFN(moe_model.ffn, DG, K, SEED)
    vq_model = LM(vq_ffn)
    # copy the trained non-FFN weights from the MoE model so we only re-heal
    vq_model.embed.load_state_dict(moe_model.embed.state_dict())
    vq_model.pos.data.copy_(moe_model.pos.data)
    vq_model.attn.load_state_dict(moe_model.attn.state_dict())
    vq_model.n1.load_state_dict(moe_model.n1.state_dict())
    vq_model.n2.load_state_dict(moe_model.n2.state_dict())
    vq_model.head.load_state_dict(moe_model.head.state_dict())
    acc0, _ = _evaluate(vq_model, X, Y)
    print(f"  VQ-MoE post-hoc  | acc {acc0:.3f}", flush=True)
    r4, _ = run("VQ-MoE+heal", vq_model, steps=400)
    rows.append(r4)

    payload = {"task": "char_lm", "chance": 0.168, "config": {
        "d_model": D_MODEL, "n_expert": N_EXPERT, "top_k": TOP_K, "expert_h": EXP_H,
        "vq_group": DG, "vq_K": K}, "vq_posthoc_acc": round(acc0, 4),
        "elapsed_sec": round(time.perf_counter() - t0, 1), "variants": rows}
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
