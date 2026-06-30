"""T12 BETTER ROUTING — does router quality decide if 90M-active reasons like the big total?

The dream rests on routing picking the RIGHT experts. This measures how much routing
quality matters: same MoE (same active compute), three routers on a capacity-bound task:
  RANDOM  : fixed random expert per token (worst)
  LEARNED : a trained router (the standard)
  ORACLE  : perfect routing (uses the true domain -> its expert) = the ceiling

If ORACLE >> LEARNED >> RANDOM, routing is the bottleneck — and the LEARNED->ORACLE gap is
the headroom that "better routing" (T12) can still capture. Same active params throughout.

Run:  python projects/v2_design/T12_routing/router_quality.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent / "router_quality_results.json"
SEED = 0
N_DOMAINS = 24
KEYS_PER = 16
VOCAB_VAL = 32
D = 48
H = 2
STEPS = 2000
BATCH = 256


def make_maps(rng):
    return [[int(rng.integers(VOCAB_VAL)) for _ in range(KEYS_PER)] for _ in range(N_DOMAINS)]


def batch(maps, rng, n):
    dom = rng.integers(N_DOMAINS, size=n); key = rng.integers(KEYS_PER, size=n)
    val = np.array([maps[dom[i]][key[i]] for i in range(n)])
    return torch.tensor(dom), torch.tensor(key), torch.tensor(val)


class MoE(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.tag = nn.Embedding(N_DOMAINS, D); self.key = nn.Embedding(KEYS_PER, D)
        self.tag.weight.requires_grad_(False); self.key.weight.requires_grad_(False)
        self.router = nn.Linear(D, N_DOMAINS)
        self.up = nn.ModuleList([nn.Linear(D, H) for _ in range(N_DOMAINS)])
        self.down = nn.ModuleList([nn.Linear(H, D) for _ in range(N_DOMAINS)])
        self.out = nn.Linear(D, VOCAB_VAL)
        # fixed random routing table (for RANDOM mode)
        g = torch.Generator().manual_seed(123)
        self.register_buffer("rand_route", torch.randint(N_DOMAINS, (N_DOMAINS,), generator=g))

    def forward(self, dom, key):
        x = self.tag(dom) + self.key(key)
        if self.mode == "oracle":
            pick = dom                                  # perfect: domain -> its expert
        elif self.mode == "random":
            pick = self.rand_route[dom]                 # fixed random expert per domain
        elif self.mode == "collapse":
            pick = torch.zeros_like(dom)                # BAD: everything -> expert 0 (overload)
        else:  # learned
            pick = self.router(x).argmax(-1)
        ff = torch.zeros_like(x)
        for e in range(N_DOMAINS):
            m = pick == e
            if m.any():
                ff[m] = self.down[e](F.gelu(self.up[e](x[m])))
        x = x + ff
        return self.out(x), (self.router(x) if self.mode == "learned" else None)


def train_eval(mode):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED); maps = make_maps(rng)
    m = MoE(mode); opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=3e-3)
    rb = np.random.default_rng(SEED + 1)
    for _ in range(STEPS):
        dom, key, val = batch(maps, rb, BATCH)
        logits, rlog = m(dom, key)
        loss = F.cross_entropy(logits, val)
        if rlog is not None:
            loss = loss + 0.5 * F.cross_entropy(rlog, dom)   # train router toward true domain
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    # eval on full deterministic set
    dom = torch.tensor(np.repeat(range(N_DOMAINS), KEYS_PER))
    key = torch.tensor(np.tile(range(KEYS_PER), N_DOMAINS))
    val = torch.tensor([maps[d][k] for d, k in zip(dom.tolist(), key.tolist())])
    with torch.no_grad():
        acc = float((m(dom, key)[0].argmax(-1) == val).float().mean())
    return acc


def main():
    print(f"capacity task: {N_DOMAINS} domains x {KEYS_PER} keys, MoE top-1 (same active)\n", flush=True)
    res = {}
    for mode in ("collapse", "random", "learned", "oracle"):
        res[mode] = round(train_eval(mode), 4)
        print(f"  {mode:8s} routing | recall {res[mode]:.3f}", flush=True)

    headroom = res["oracle"] - res["learned"]
    print(f"\n  random {res['random']:.3f} | learned {res['learned']:.3f} | ORACLE {res['oracle']:.3f}",
          flush=True)
    print(f"  routing matters: oracle - random = {res['oracle']-res['random']:+.3f}", flush=True)
    print(f"  HEADROOM (learned -> oracle) = {headroom:+.3f}  <- what 'better routing' can still win",
          flush=True)
    verdict = ("routing is THE bottleneck: perfect routing >> learned >> random; closing the "
               "learned->oracle gap is the T12 prize" if res["oracle"] > res["random"] + 0.2
               else "routing matters little on this task")
    print(f"  VERDICT: {verdict}", flush=True)

    OUT.write_text(json.dumps({"random": res["random"], "learned": res["learned"],
                   "oracle": res["oracle"], "headroom": round(headroom, 4), "verdict": verdict,
                   "note": "same MoE / same active params, three routers. oracle=ceiling, "
                           "learned=current, random=floor. learned->oracle gap = the headroom "
                           "better routing (T12) can still capture toward full-total quality."},
                   indent=2), encoding="utf-8")
    print(f"\nwritten {OUT}", flush=True)


if __name__ == "__main__":
    main()
