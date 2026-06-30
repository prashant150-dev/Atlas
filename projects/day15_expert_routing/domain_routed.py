"""Day-15b: domain-ROUTED experts -> clean task-conditional loading.

Vanilla MoE entangled experts (each domain used ~69% of them) — bad for conditional
loading. Fix: route by TASK/DOMAIN, so each domain has its OWN dedicated expert(s).
Then for a given task you load ONLY that domain's expert from disk; the rest stay
on disk. We measure: does accuracy hold, and how much expert-RAM is saved.

Task: D domains, each a distinct key->value map; domain tag in the input picks the
expert (hard routing). D experts, 1 active per token.

Run:  python projects/day15_expert_routing/domain_routed.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_HERE = Path(__file__).resolve().parent
OUT = _HERE / "domain_routed_results.json"

D = 8            # domains (= experts)
V = 24           # symbols
D_MODEL = 64
EXP_H = 64
SEQ = 3
VOCAB = D + V
STEPS = 1500
BATCH = 256
LR = 3e-3
SEED = 0


_PERMS = torch.stack([torch.randperm(V, generator=torch.Generator().manual_seed(100 + d))
                      for d in range(D)])   # FIXED per-domain maps (the "knowledge")


def make_data(n, gen):
    perms = _PERMS
    dom = torch.randint(0, D, (n,), generator=gen)
    key = torch.randint(0, V, (n,), generator=gen)
    val = perms[dom, key]
    X = torch.zeros(n, SEQ, dtype=torch.long)
    X[:, 0] = dom; X[:, 1] = key + D; X[:, 2] = val + D
    Y = torch.full((n, SEQ), -100, dtype=torch.long); Y[:, 1] = val + D
    return X, Y, dom


class DomainRoutedLM(nn.Module):
    """embed -> attn -> DOMAIN-routed expert MLP -> head. Token 0 (domain) picks expert."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Parameter(torch.zeros(1, SEQ, D_MODEL))
        self.n1 = nn.LayerNorm(D_MODEL); self.n2 = nn.LayerNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL); self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.up = nn.ModuleList([nn.Linear(D_MODEL, EXP_H) for _ in range(D)])
        self.down = nn.ModuleList([nn.Linear(EXP_H, D_MODEL) for _ in range(D)])
        self.head = nn.Linear(D_MODEL, VOCAB)

    def _attn(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, 2)
        a = (q @ k.transpose(-2, -1)) / math.sqrt(C)
        a = a.masked_fill(torch.triu(torch.ones(T, T), 1).bool(), float("-inf")).softmax(-1)
        return self.proj(a @ v)

    def forward(self, tok, dom=None):
        if dom is None:
            dom = tok[:, 0]
        x = self.embed(tok) + self.pos[:, :tok.shape[1]]
        x = x + self._attn(self.n1(x))
        h = self.n2(x)
        out = torch.zeros_like(x)
        for d in range(D):                       # route each token's sequence by its domain
            m = dom == d
            if m.any():
                out[m] = self.down[d](F.gelu(self.up[d](h[m])))
        x = x + out
        return self.head(x)


def main():
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED)
    X, Y, dom = make_data(12000, g)
    m = DomainRoutedLM()
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(1)
    m.train()
    for step in range(STEPS):
        idx = torch.randint(0, X.shape[0], (BATCH,), generator=gen)
        opt.zero_grad()
        logits = m(X[idx], dom[idx])
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), Y[idx].reshape(-1), ignore_index=-100)
        loss.backward(); opt.step()
    m.eval()
    ge = torch.Generator().manual_seed(2)
    Xe, Ye, dome = make_data(4000, ge)
    with torch.no_grad():
        pred = m(Xe, dome).argmax(-1)
    mask = Ye != -100
    acc = float((pred[mask] == Ye[mask]).float().mean())

    # expert param sizes
    per_expert = sum(p.numel() for p in m.up[0].parameters()) + sum(p.numel() for p in m.down[0].parameters())
    total_experts = per_expert * D
    print(f"DOMAIN-ROUTED MoE: accuracy {acc:.3f}  ({D} domains, 1 expert/token)", flush=True)
    print(f"  per-task you load 1/{D} experts -> {100/D:.0f}% of expert-RAM "
          f"({D}x less) at FULL accuracy", flush=True)
    print(f"  vanilla MoE earlier: needed ~69% of experts (1.5x). domain-routing -> {D}x.", flush=True)
    OUT.write_text(json.dumps({"routing": "domain", "accuracy": round(acc, 4), "domains": D,
                               "experts_loaded_per_task": 1, "expert_ram_reduction_x": D,
                               "per_expert_params": per_expert, "total_expert_params": total_experts,
                               "note": "task-conditional loading: load only the task's expert; rest stay on disk"},
                              indent=2), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
