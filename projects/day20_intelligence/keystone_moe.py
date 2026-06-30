"""Part-4 BEAST INTELLIGENCE — 4A keystone: does sparse ACTIVE capacity reason like
the big TOTAL, or like its small active size?

The whole dream rests on one assumption: a model with ~80-100M ACTIVE params (out of a
400B-on-disk TOTAL) can be beast-smart — i.e. sparsity buys the intelligence of the big
total at the compute of the small active set. This experiment measures that directly,
at small scale, on a task where TOTAL capacity is what matters.

TASK (capacity-bound associative recall): there are N_DOMAINS disjoint key->value maps.
Each sequence is tagged with a domain and asks to recall values for keys IN THAT DOMAIN.
To score well a model must STORE all N domains' maps (total capacity), but any single
forward pass only needs ONE domain (small active compute). This is exactly the regime
where MoE should shine: experts specialise per domain, router selects.

Three models compared at EQUAL ACTIVE COMPUTE per token:
  * dense_small : hidden = h         (active = total = small;  can it hold all domains?)
  * moe         : N experts of h, top-1 (active = h, total = N*h; router picks domain)
  * dense_big   : hidden = N*h       (active = total = big;  the quality CEILING)

Thesis (dream holds) iff:  moe  >>  dense_small   and   moe ~~ dense_big
i.e. sparse active capacity reasons like the big total, not like its small active size.

Run:  python projects/day20_intelligence/keystone_moe.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "keystone_results.json"

SEED = 0
N_DOMAINS = 32          # disjoint key->value maps the model must store
KEYS_PER = 32           # keys in each domain
VOCAB_KEY = KEYS_PER    # key embeddings are SHARED across domains (forces FFN to compute
                        # the mapping, not memorise unique per-(domain,key) embeddings)
VOCAB_VAL = 64          # value alphabet (6 bits/fact); total facts = 32*32 = 1024
D_MODEL = 32
H = 4                   # tiny ACTIVE hidden = the capacity bottleneck knob
N_EXPERT = N_DOMAINS    # one expert capacity slot per domain (router must learn routing)
STEPS = 3000
BATCH = 256
LR = 3e-3


def _make_maps(rng):
    """N_DOMAINS disjoint maps: key-id -> value-id. Fixed across train/eval."""
    maps = []
    for d in range(N_DOMAINS):
        vals = [int(rng.integers(VOCAB_VAL)) for _ in range(KEYS_PER)]
        maps.append(vals)
    return maps


def _batch(maps, rng, n):
    """input = [domain_tag, key]; target = value. domain_tag in [0,N), key in domain."""
    dom = rng.integers(N_DOMAINS, size=n)
    key = rng.integers(KEYS_PER, size=n)
    tags = torch.tensor(dom, dtype=torch.long)
    keys = torch.tensor(key, dtype=torch.long)                   # LOCAL key id (shared)
    vals = torch.tensor([maps[dom[i]][key[i]] for i in range(n)], dtype=torch.long)
    return tags, keys, vals


class _Embed(nn.Module):
    """FROZEN random embeddings — the model cannot memorise the answer in the embedding
    table; it must use FFN / expert capacity to map (domain,key) -> value."""

    def __init__(self):
        super().__init__()
        self.tag = nn.Embedding(N_DOMAINS, D_MODEL)
        self.key = nn.Embedding(VOCAB_KEY, D_MODEL)
        self.tag.weight.requires_grad_(False)
        self.key.weight.requires_grad_(False)

    def forward(self, tags, keys):
        return self.tag(tags) + self.key(keys)            # [B, D]


class DenseHead(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.emb = _Embed()
        self.up = nn.Linear(D_MODEL, h)
        self.down = nn.Linear(h, D_MODEL)
        self.out = nn.Linear(D_MODEL, VOCAB_VAL)
        self.h = h

    def forward(self, tags, keys):
        x = self.emb(tags, keys)
        x = x + self.down(F.gelu(self.up(x)))
        return self.out(x)

    def active_params(self):
        return self.up.weight.numel() + self.down.weight.numel()


class MoEHead(nn.Module):
    def __init__(self, n_expert, h, top_k=1):
        super().__init__()
        self.emb = _Embed()
        self.router = nn.Linear(D_MODEL, n_expert)
        self.up = nn.ModuleList([nn.Linear(D_MODEL, h) for _ in range(n_expert)])
        self.down = nn.ModuleList([nn.Linear(h, D_MODEL) for _ in range(n_expert)])
        self.out = nn.Linear(D_MODEL, VOCAB_VAL)
        self.n_expert, self.top_k, self.h = n_expert, top_k, h

    def forward(self, tags, keys):
        x = self.emb(tags, keys)
        scores = self.router(x)
        topv, topi = scores.topk(self.top_k, dim=-1)
        gate = topv.softmax(-1)
        ff = torch.zeros_like(x)
        for slot in range(self.top_k):
            idx = topi[:, slot]; g = gate[:, slot:slot + 1]
            for e in range(self.n_expert):
                m = idx == e
                if m.any():
                    ff[m] += g[m] * self.down[e](F.gelu(self.up[e](x[m])))
        x = x + ff
        return self.out(x)

    def active_params(self):
        per = self.up[0].weight.numel() + self.down[0].weight.numel()
        return per * self.top_k + self.router.weight.numel()


def _train_eval(model, maps, name):
    rng = __import__("numpy").random.default_rng(SEED + 1)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for step in range(STEPS):
        tags, keys, vals = _batch(maps, rng, BATCH)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(tags, keys), vals)
        loss.backward(); opt.step()
    # eval on the FULL deterministic key set (every domain, every key)
    model.eval()
    with torch.no_grad():
        dom = __import__("numpy").repeat(range(N_DOMAINS), KEYS_PER)
        kk = __import__("numpy").tile(range(KEYS_PER), N_DOMAINS)
        tags = torch.tensor(dom, dtype=torch.long)
        keys = torch.tensor(kk, dtype=torch.long)               # LOCAL key id (shared)
        vals = torch.tensor([maps[dom[i]][kk[i]] for i in range(len(dom))], dtype=torch.long)
        pred = model(tags, keys).argmax(-1)
        acc = float((pred == vals).float().mean())
    tot = sum(p.numel() for p in model.parameters())
    print(f"  {name:12s} | active {model.active_params():6d} | total params {tot:7d} | "
          f"recall {acc:.3f}", flush=True)
    return {"name": name, "active_ffn": model.active_params(), "total_params": tot,
            "recall": round(acc, 4)}


def main():
    torch.manual_seed(SEED)
    import numpy as np
    rng = np.random.default_rng(SEED)
    maps = _make_maps(rng)
    print(f"task: {N_DOMAINS} disjoint maps x {KEYS_PER} keys = {VOCAB_KEY} facts to store; "
          f"one domain active per query\n", flush=True)

    rows = []
    # dense_small: active == one expert's hidden H (the small-active baseline)
    rows.append(_train_eval(DenseHead(H), maps, "dense_small"))
    # moe: N experts of H, top-1 -> active == H + router, total ~ N*H (the dream's regime)
    moe = MoEHead(N_EXPERT, H, top_k=1)
    rows.append(_train_eval(moe, maps, "moe_top1"))
    # dense_MATCH control: a DENSE model with the SAME active params as the MoE (router
    # included). If it still fails while MoE wins, the lever is TOTAL capacity (sparsity),
    # not active compute. H chosen so 2*D*H ~= moe active params.
    h_match = max(H + 1, round(moe.active_params() / (2 * D_MODEL)))
    rows.append(_train_eval(DenseHead(h_match), maps, f"dense_match(h={h_match})"))
    # dense_big: hidden = N*H -> active == total == big (the quality ceiling)
    rows.append(_train_eval(DenseHead(N_EXPERT * H), maps, "dense_big"))

    small = next(r for r in rows if r["name"] == "dense_small")["recall"]
    moe = next(r for r in rows if r["name"] == "moe_top1")["recall"]
    big = next(r for r in rows if r["name"] == "dense_big")["recall"]
    verdict = ("DREAM-CONSISTENT: sparse active reasons like the big total"
               if moe >= 0.9 * big and moe > small + 0.15 else
               "NOT shown: sparsity did not buy the big-total capacity here")
    print(f"\n  dense_small {small:.3f} | moe {moe:.3f} | dense_big {big:.3f}", flush=True)
    print(f"  active(moe)~active(dense_small); total(moe)~total(dense_big)", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)

    payload = {"n_domains": N_DOMAINS, "keys_per": KEYS_PER, "H": H, "n_expert": N_EXPERT,
               "steps": STEPS, "rows": rows,
               "dense_small": small, "moe": moe, "dense_big": big, "verdict": verdict,
               "note": "4A keystone: at equal ACTIVE compute, does MoE (small active, big "
                       "total) match dense_big and beat dense_small on a capacity-bound task? "
                       "If yes, the dream's 'sparse active is smart' assumption holds."}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten {OUT}", flush=True)


if __name__ == "__main__":
    main()
