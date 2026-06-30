"""T1 SIZE, Lever 2b — NATIVE sparse training vs POST-HOC pruning at EXTREME sparsity.

The honest reframe: 0.15 bits/weight is impossible PER-WEIGHT (Day-1: entropy ~2 bits),
but reachable as an AVERAGE if ~96% of weights are ZERO (Day-16). Zeros cost ~0 bits.
So the real research question is: CAN A MODEL BE 90-98% SPARSE AND STILL WORK?

We saw post-hoc pruning collapse (Day lever2: 50% sparse -> dead). This tests whether
NATIVE sparse training (the mask is fixed BEFORE training, weights learn around it)
survives where post-hoc dies. If native >> post-hoc at extreme sparsity, the path to
sub-0.15-bit-average is native training (brains are ~99% sparse; lottery-ticket thesis).

Task: a learnable mapping (capacity-bound) so sparsity actually bites. We compare at
each sparsity s in {90,95,98%}:
  * dense          : full model (ceiling)
  * post-hoc prune : train dense, then zero the smallest |w|, no retrain (the collapse)
  * native sparse  : fix a random s-sparse mask, train WITH it from scratch

Run:  python projects/v2_design/T1_size/lever2b_native_sparse.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent / "lever2b_results.json"
SEED = 0
# task: predict y = f(x) where f is a fixed random nonlinear map (needs capacity)
IN, HID, OUT_D, N = 32, 512, 16, 4000
STEPS = 1500
BATCH = 256


def _make_task(rng):
    # teacher: a fixed 2-layer random net -> target classes; student must match it
    W1 = rng.standard_normal((IN, 256)).astype(np.float32) / np.sqrt(IN)
    W2 = rng.standard_normal((256, OUT_D)).astype(np.float32) / np.sqrt(256)
    X = rng.standard_normal((N, IN)).astype(np.float32)
    H = np.maximum(X @ W1, 0)
    Y = (H @ W2).argmax(1)
    return torch.tensor(X), torch.tensor(Y)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IN, HID)
        self.fc2 = nn.Linear(HID, OUT_D)
        self.register_buffer("m1", torch.ones_like(self.fc1.weight))
        self.register_buffer("m2", torch.ones_like(self.fc2.weight))

    def set_mask(self, s, rng):
        for m in (self.m1, self.m2):
            flat = m.view(-1)
            k = int(len(flat) * s)
            idx = torch.tensor(rng.choice(len(flat), k, replace=False))
            flat.fill_(1.0); flat[idx] = 0.0

    def forward(self, x):
        h = F.relu(F.linear(x, self.fc1.weight * self.m1, self.fc1.bias))
        return F.linear(h, self.fc2.weight * self.m2, self.fc2.bias)


def _train(model, X, Y, steps=STEPS):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(SEED)
    model.train()
    for _ in range(steps):
        i = torch.randint(len(X), (BATCH,), generator=g)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(X[i]), Y[i]).backward()
        opt.step()


@torch.no_grad()
def _acc(model, X, Y):
    return float((model(X).argmax(1) == Y).float().mean())


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    X, Y = _make_task(rng)
    ntr = int(0.8 * N)
    Xtr, Ytr, Xte, Yte = X[:ntr], Y[:ntr], X[ntr:], Y[ntr:]

    # dense ceiling
    dense = Net(); _train(dense, Xtr, Ytr)
    dense_acc = _acc(dense, Xte, Yte)
    print(f"dense (0% sparse) held-out acc: {dense_acc:.3f}\n", flush=True)

    rows = [{"sparsity": 0.0, "mode": "dense", "acc": round(dense_acc, 4)}]
    print(f"{'sparsity':>8} | {'post-hoc prune':>14} | {'NATIVE sparse':>13}", flush=True)
    for s in (0.90, 0.95, 0.98):
        # post-hoc: take the trained dense, zero smallest |w| per layer, NO retrain
        ph = Net(); ph.load_state_dict(dense.state_dict())
        for w, m in ((ph.fc1.weight, ph.m1), (ph.fc2.weight, ph.m2)):
            flat = w.detach().abs().view(-1)
            k = int(len(flat) * s)
            thresh = flat.kthvalue(k).values
            m.copy_((w.detach().abs() > thresh).float())
        ph_acc = _acc(ph, Xte, Yte)

        # native: fix an s-sparse random mask, train WITH it from scratch
        nat = Net(); nat.set_mask(s, np.random.default_rng(SEED + 7)); _train(nat, Xtr, Ytr)
        nat_acc = _acc(nat, Xte, Yte)

        rows.append({"sparsity": s, "post_hoc_acc": round(ph_acc, 4),
                     "native_acc": round(nat_acc, 4)})
        print(f"{s*100:6.0f}% | {ph_acc:14.3f} | {nat_acc:13.3f}", flush=True)

    print("\nHonest read:", flush=True)
    best = rows[-1]
    print(f"  at 98% sparse (~0.32 effective bits if ternary): post-hoc {best['post_hoc_acc']:.2f} "
          f"vs NATIVE {best['native_acc']:.2f} (dense {dense_acc:.2f})", flush=True)
    verdict = ("NATIVE sparse SURVIVES where post-hoc dies -> the path to ultra-low avg-bits "
               "is native training" if best["native_acc"] > best["post_hoc_acc"] + 0.1
               else "native did not clearly beat post-hoc here — needs better sparse training")
    print(f"  VERDICT: {verdict}", flush=True)
    OUT.write_text(json.dumps({"dense_acc": dense_acc, "rows": rows, "verdict": verdict,
                   "note": "native sparse training vs post-hoc pruning at extreme sparsity; "
                           "tests whether sub-0.15-bit-AVERAGE (via 96%+ zeros) is reachable "
                           "natively (post-hoc collapses)."}, indent=2), encoding="utf-8")
    print(f"\nwritten {OUT}", flush=True)


if __name__ == "__main__":
    main()
