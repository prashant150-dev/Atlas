"""T1 SIZE, Lever 2c — SMART (learned) sparse mask vs random, at extreme sparsity.

Lever-2b: native-sparse (RANDOM mask) >> post-hoc. Now the obvious upgrade: don't keep a
RANDOM 10% of weights — keep the IMPORTANT ones. The lottery-ticket recipe:
  1. train a dense model,
  2. find the top-(1-s) weights BY MAGNITUDE (the ones that mattered),
  3. fix that mask and TRAIN A FRESH sparse model with it (native, informed).

Compare at each sparsity: dense (ceiling) · post-hoc (dead) · random-native (2b) · SMART-native.
If smart-native > random-native, we have a better lever toward ultra-low average bits.

Run:  python projects/v2_design/T1_size/lever2c_smart_mask.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent / "lever2c_results.json"
SEED = 0
IN, HID, OUT_D, N = 32, 512, 16, 4000
STEPS = 1500
BATCH = 256


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

    def random_mask(self, s, rng):
        for m in (self.m1, self.m2):
            f = m.view(-1); k = int(len(f) * s)
            idx = torch.tensor(rng.choice(len(f), k, replace=False))
            f.fill_(1.0); f[idx] = 0.0

    def smart_mask_from(self, donor, s):
        """keep the top-(1-s) weights BY MAGNITUDE from a trained donor model."""
        for m, w in ((self.m1, donor.fc1.weight), (self.m2, donor.fc2.weight)):
            f = w.detach().abs().view(-1); k = int(len(f) * s)
            thr = f.kthvalue(k).values
            m.copy_((w.detach().abs() > thr).float())

    def forward(self, x):
        h = F.relu(F.linear(x, self.fc1.weight * self.m1, self.fc1.bias))
        return F.linear(h, self.fc2.weight * self.m2, self.fc2.bias)


def _train(model, X, Y):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(SEED)
    model.train()
    for _ in range(STEPS):
        i = torch.randint(len(X), (BATCH,), generator=g)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(X[i]), Y[i]).backward()
        opt.step()


@torch.no_grad()
def _acc(m, X, Y):
    return float((m(X).argmax(1) == Y).float().mean())


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    X, Y = _make_task(rng)
    ntr = int(0.8 * N); Xtr, Ytr, Xte, Yte = X[:ntr], Y[:ntr], X[ntr:], Y[ntr:]

    dense = Net(); _train(dense, Xtr, Ytr)
    dense_acc = _acc(dense, Xte, Yte)
    print(f"dense acc {dense_acc:.3f}\n", flush=True)
    print(f"{'sparsity':>8} | {'post-hoc':>9} | {'random-native':>13} | {'SMART-native':>12}", flush=True)

    rows = [{"sparsity": 0.0, "dense": round(dense_acc, 4)}]
    for s in (0.90, 0.95, 0.98):
        # post-hoc
        ph = Net(); ph.load_state_dict(dense.state_dict()); ph.smart_mask_from(dense, s)
        ph_acc = _acc(ph, Xte, Yte)
        # random native
        rn = Net(); rn.random_mask(s, np.random.default_rng(SEED + 7)); _train(rn, Xtr, Ytr)
        rn_acc = _acc(rn, Xte, Yte)
        # SMART native: lottery-ticket mask (top-|w| from dense) + fresh sparse training
        sm = Net(); sm.smart_mask_from(dense, s); _train(sm, Xtr, Ytr)
        sm_acc = _acc(sm, Xte, Yte)
        rows.append({"sparsity": s, "post_hoc": round(ph_acc, 4),
                     "random_native": round(rn_acc, 4), "smart_native": round(sm_acc, 4)})
        print(f"{s*100:6.0f}% | {ph_acc:9.3f} | {rn_acc:13.3f} | {sm_acc:12.3f}", flush=True)

    best = rows[-1]
    print(f"\nat 98% sparse (only 2% weights): dense {dense_acc:.2f} | post-hoc {best['post_hoc']:.2f} "
          f"| random {best['random_native']:.2f} | SMART {best['smart_native']:.2f}", flush=True)
    gain = best["smart_native"] - best["random_native"]
    verdict = (f"SMART mask beats random by +{gain:.2f} at 98% -> informed native sparsity is the "
               f"better lever" if gain > 0.03 else "smart ~= random here (task too easy or mask weak)")
    print(f"VERDICT: {verdict}", flush=True)
    OUT.write_text(json.dumps({"dense_acc": dense_acc, "rows": rows, "verdict": verdict,
                   "note": "smart (lottery-ticket magnitude) mask vs random mask, both native; "
                           "post-hoc shown dead. Tests the better mask for ultra-sparse."},
                   indent=2), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
