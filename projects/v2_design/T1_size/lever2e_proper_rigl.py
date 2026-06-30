"""T1 SIZE, Lever 2e — PROPER RigL (gradient-based regrowth) to break the 59% ceiling.

Lever-2d failed because my "iterative" only re-ranked by MAGNITUDE — a zeroed weight stays
zero forever (its usefulness is never seen). PROPER RigL fixes this:

  every K steps:
    DROP : among ACTIVE weights, kill the fraction with smallest |weight|.
    GROW : among INACTIVE (zero) weights, activate those with the largest |GRADIENT|
           (computed on the DENSE weight) — i.e. connections that WOULD reduce loss if
           they existed. Newly grown weights start at 0.
  drop fraction decays (cosine) so the mask settles by the end.

This lets the sparse model EXPLORE which 2% of weights to keep, guided by gradients, not
just current magnitude. Compare vs fixed-smart (2d best 0.384) and dense ceiling.

Run:  python projects/v2_design/T1_size/lever2e_proper_rigl.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent / "lever2e_results.json"
SEED = 0
IN, HID, OUT_D, N = 32, 512, 16, 4000
STEPS = 2000
BATCH = 256
TARGET_S = 0.98
RIGL_EVERY = 100
DROP_FRAC0 = 0.3


def _make_task(rng):
    W1 = rng.standard_normal((IN, 256)).astype(np.float32) / np.sqrt(IN)
    W2 = rng.standard_normal((256, OUT_D)).astype(np.float32) / np.sqrt(256)
    X = rng.standard_normal((N, IN)).astype(np.float32)
    Y = (np.maximum(X @ W1, 0) @ W2).argmax(1)
    return torch.tensor(X), torch.tensor(Y)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IN, HID)
        self.fc2 = nn.Linear(HID, OUT_D)
        self.register_buffer("m1", torch.ones_like(self.fc1.weight))
        self.register_buffer("m2", torch.ones_like(self.fc2.weight))

    def init_mask(self, s):
        for m, w in ((self.m1, self.fc1.weight), (self.m2, self.fc2.weight)):
            f = w.detach().abs().view(-1); k = int(len(f) * s)
            m.copy_((w.detach().abs() > f.kthvalue(k).values).float())

    def forward(self, x, masked=True):
        w1 = self.fc1.weight * self.m1 if masked else self.fc1.weight
        w2 = self.fc2.weight * self.m2 if masked else self.fc2.weight
        return F.linear(F.relu(F.linear(x, w1, self.fc1.bias)), w2, self.fc2.bias)


@torch.no_grad()
def _acc(m, X, Y):
    return float((m(X).argmax(1) == Y).float().mean())


def _rigl_update(model, X, Y, drop_frac, gen):
    """one DROP+GROW mask migration using dense gradients."""
    i = torch.randint(len(X), (BATCH,), generator=gen)
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(X[i], masked=False), Y[i])    # DENSE forward -> all grads
    loss.backward()
    for w, m in ((model.fc1.weight, model.m1), (model.fc2.weight, model.m2)):
        grad = w.grad.detach().abs()
        mask = m.bool()
        n_active = int(mask.sum())
        n_move = int(drop_frac * n_active)
        if n_move < 1:
            continue
        wabs = w.detach().abs()
        # DROP: smallest-|w| among active
        active_idx = mask.view(-1).nonzero(as_tuple=True)[0]
        drop_local = wabs.view(-1)[active_idx].topk(n_move, largest=False).indices
        drop_idx = active_idx[drop_local]
        # GROW: largest-|grad| among inactive
        inactive_idx = (~mask.view(-1)).nonzero(as_tuple=True)[0]
        grow_local = grad.view(-1)[inactive_idx].topk(n_move, largest=True).indices
        grow_idx = inactive_idx[grow_local]
        mflat = m.view(-1); wflat = w.detach().view(-1)
        mflat[drop_idx] = 0.0
        mflat[grow_idx] = 1.0
        wflat[grow_idx] = 0.0                                  # grown weights start at 0
    model.zero_grad(set_to_none=True)


def train_rigl(X, Y):
    m = Net(); m.init_mask(TARGET_S)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(SEED)
    for step in range(STEPS):
        if step and step % RIGL_EVERY == 0 and step < STEPS * 0.8:
            df = DROP_FRAC0 * 0.5 * (1 + math.cos(math.pi * step / (STEPS * 0.8)))
            _rigl_update(m, X, Y, df, g)
        i = torch.randint(len(X), (BATCH,), generator=g)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(m(X[i]), Y[i]).backward()
        # keep grads only on active weights
        m.fc1.weight.grad.mul_(m.m1); m.fc2.weight.grad.mul_(m.m2)
        opt.step()
    return m


def train_fixed_smart(X, Y, donor):
    m = Net()
    for mm, w in ((m.m1, donor.fc1.weight), (m.m2, donor.fc2.weight)):
        f = w.detach().abs().view(-1); k = int(len(f) * TARGET_S)
        mm.copy_((w.detach().abs() > f.kthvalue(k).values).float())
    opt = torch.optim.Adam(m.parameters(), lr=3e-3); g = torch.Generator().manual_seed(SEED)
    for _ in range(STEPS):
        i = torch.randint(len(X), (BATCH,), generator=g)
        opt.zero_grad(set_to_none=True); F.cross_entropy(m(X[i]), Y[i]).backward()
        m.fc1.weight.grad.mul_(m.m1); m.fc2.weight.grad.mul_(m.m2); opt.step()
    return m


def train_dense(X, Y):
    m = Net(); opt = torch.optim.Adam(m.parameters(), lr=3e-3); g = torch.Generator().manual_seed(SEED)
    for _ in range(STEPS):
        i = torch.randint(len(X), (BATCH,), generator=g)
        opt.zero_grad(set_to_none=True); F.cross_entropy(m(X[i]), Y[i]).backward(); opt.step()
    return m


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    X, Y = _make_task(rng)
    ntr = int(0.8 * N); Xtr, Ytr, Xte, Yte = X[:ntr], Y[:ntr], X[ntr:], Y[ntr:]

    dense = train_dense(Xtr, Ytr); d = _acc(dense, Xte, Yte)
    fixed = train_fixed_smart(Xtr, Ytr, dense); fx = _acc(fixed, Xte, Yte)
    rigl = train_rigl(Xtr, Ytr); rg = _acc(rigl, Xte, Yte)

    print(f"all at {TARGET_S*100:.0f}% sparse (2% weights), dense {d:.3f}\n", flush=True)
    print(f"  fixed-smart      : {fx:.3f}  ({fx/d*100:.0f}% of dense)", flush=True)
    print(f"  PROPER RigL      : {rg:.3f}  ({rg/d*100:.0f}% of dense)", flush=True)
    gain = rg - fx
    verdict = (f"RigL BREAKS the ceiling: +{gain:.3f} over fixed-smart" if gain > 0.03
               else f"RigL did not beat fixed-smart (+{gain:.3f}) — ceiling holds on this task")
    print(f"\n  VERDICT: {verdict}", flush=True)
    OUT.write_text(json.dumps({"dense": round(d, 4), "fixed_smart": round(fx, 4),
                   "proper_rigl": round(rg, 4), "gain": round(gain, 4), "verdict": verdict,
                   "note": "proper RigL (gradient regrowth) vs fixed-smart at 98% sparse; "
                           "tries to break the ~59%-of-dense ceiling."}, indent=2),
                   encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
