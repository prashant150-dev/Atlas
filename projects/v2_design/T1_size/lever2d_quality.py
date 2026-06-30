"""T1 SIZE, Lever 2d — push the QUALITY of an ultra-sparse model UP (59% -> higher).

Lever-2c: SMART fixed mask gave 0.384 at 98% sparse (dense 0.647). Now improve the
TRAINING so the sparse model reaches higher quality. Two well-known levers:

  * GRADUAL sparsification: don't slam to 98% at step 0 — ramp sparsity 0 -> 98% over
    training (cubic schedule) so weights ADAPT smoothly as the budget tightens.
  * ITERATIVE mask (RigL-lite): keep a fixed 98% budget but RE-CHOOSE which weights to
    keep every K steps (drop the now-smallest, let better weights take their place).

Compare all at 98% sparse: dense ceiling · fixed-smart (2c) · gradual · iterative.

Run:  python projects/v2_design/T1_size/lever2d_quality.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent / "lever2d_results.json"
SEED = 0
IN, HID, OUT_D, N = 32, 512, 16, 4000
STEPS = 2000
BATCH = 256
TARGET_S = 0.98


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

    def apply_magnitude_mask(self, s):
        """keep top-(1-s) by current |w| per layer."""
        for m, w in ((self.m1, self.fc1.weight), (self.m2, self.fc2.weight)):
            f = w.detach().abs().view(-1)
            k = int(len(f) * s)
            if k <= 0:
                m.fill_(1.0); continue
            thr = f.kthvalue(k).values
            m.copy_((w.detach().abs() > thr).float())

    def forward(self, x):
        h = F.relu(F.linear(x, self.fc1.weight * self.m1, self.fc1.bias))
        return F.linear(h, self.fc2.weight * self.m2, self.fc2.bias)


def _opt(model):
    return torch.optim.Adam(model.parameters(), lr=3e-3)


@torch.no_grad()
def _acc(m, X, Y):
    return float((m(X).argmax(1) == Y).float().mean())


def train_dense(X, Y):
    m = Net(); opt = _opt(m); g = torch.Generator().manual_seed(SEED)
    for _ in range(STEPS):
        i = torch.randint(len(X), (BATCH,), generator=g)
        opt.zero_grad(set_to_none=True); F.cross_entropy(m(X[i]), Y[i]).backward(); opt.step()
    return m


def train_fixed_smart(X, Y, donor):
    m = Net()
    # smart mask from donor
    for mm, w in ((m.m1, donor.fc1.weight), (m.m2, donor.fc2.weight)):
        f = w.detach().abs().view(-1); k = int(len(f) * TARGET_S)
        mm.copy_((w.detach().abs() > f.kthvalue(k).values).float())
    opt = _opt(m); g = torch.Generator().manual_seed(SEED)
    for _ in range(STEPS):
        i = torch.randint(len(X), (BATCH,), generator=g)
        opt.zero_grad(set_to_none=True); F.cross_entropy(m(X[i]), Y[i]).backward(); opt.step()
    return m


def train_gradual(X, Y, update_every=50):
    """ramp sparsity 0 -> TARGET_S over training (cubic), re-mask by magnitude."""
    m = Net(); opt = _opt(m); g = torch.Generator().manual_seed(SEED)
    for step in range(STEPS):
        frac = min(1.0, step / (STEPS * 0.8))            # reach target at 80% of training
        s = TARGET_S * (1 - (1 - frac) ** 3)             # cubic ramp
        if step % update_every == 0:
            m.apply_magnitude_mask(s)
        i = torch.randint(len(X), (BATCH,), generator=g)
        opt.zero_grad(set_to_none=True); F.cross_entropy(m(X[i]), Y[i]).backward(); opt.step()
    m.apply_magnitude_mask(TARGET_S)
    return m


def train_iterative(X, Y, update_every=100):
    """fixed TARGET_S budget but RE-CHOOSE the kept weights every K steps (RigL-lite)."""
    m = Net(); m.apply_magnitude_mask(TARGET_S); opt = _opt(m)
    g = torch.Generator().manual_seed(SEED)
    for step in range(STEPS):
        if step and step % update_every == 0:
            m.apply_magnitude_mask(TARGET_S)             # migrate the mask to current best
        i = torch.randint(len(X), (BATCH,), generator=g)
        opt.zero_grad(set_to_none=True); F.cross_entropy(m(X[i]), Y[i]).backward(); opt.step()
    m.apply_magnitude_mask(TARGET_S)
    return m


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    X, Y = _make_task(rng)
    ntr = int(0.8 * N); Xtr, Ytr, Xte, Yte = X[:ntr], Y[:ntr], X[ntr:], Y[ntr:]

    dense = train_dense(Xtr, Ytr); dense_acc = _acc(dense, Xte, Yte)
    fixed = train_fixed_smart(Xtr, Ytr, dense); fixed_acc = _acc(fixed, Xte, Yte)
    grad = train_gradual(Xtr, Ytr); grad_acc = _acc(grad, Xte, Yte)
    itr = train_iterative(Xtr, Ytr); itr_acc = _acc(itr, Xte, Yte)

    print(f"all at {TARGET_S*100:.0f}% sparse (2% weights), dense ceiling {dense_acc:.3f}\n", flush=True)
    print(f"  fixed-smart  (2c) : {fixed_acc:.3f}", flush=True)
    print(f"  GRADUAL ramp      : {grad_acc:.3f}", flush=True)
    print(f"  ITERATIVE (RigL)  : {itr_acc:.3f}", flush=True)
    best = max(grad_acc, itr_acc, fixed_acc)
    print(f"\n  best sparse {best:.3f} = {best/dense_acc*100:.0f}% of dense "
          f"(was 59% with fixed-smart)", flush=True)
    gain = best - fixed_acc
    print(f"  improvement over fixed-smart: +{gain:.3f}", flush=True)

    OUT.write_text(json.dumps({"target_sparsity": TARGET_S, "dense": round(dense_acc, 4),
                   "fixed_smart": round(fixed_acc, 4), "gradual": round(grad_acc, 4),
                   "iterative": round(itr_acc, 4), "best_frac_of_dense": round(best/dense_acc, 3),
                   "note": "quality-boost levers at 98% sparse: gradual ramp + iterative mask "
                           "vs fixed-smart. Pushes the ultra-sparse quality toward dense."},
                   indent=2), encoding="utf-8")
    print(f"\nwritten {OUT}", flush=True)


if __name__ == "__main__":
    main()
