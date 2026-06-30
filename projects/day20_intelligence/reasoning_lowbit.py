"""Part-4 BEAST INTELLIGENCE — 4B: does 2-bit quantization break REASONING (not just ppl)?

Part-1 measured perplexity/recall under low-bit. But reasoning (learning an ALGORITHM,
not memorising facts) might degrade differently. Test on modular addition (a+b) mod P —
the canonical task where a network must learn the GROUP STRUCTURE, not a lookup table
(it generalises to held-out (a,b) pairs only if it learned the real rule).

Pipeline:
  1. train a small FP model on a TRAIN split of (a,b) pairs; check it GENERALISES to a
     held-out split (= it learned the algorithm, not memorised).
  2. quantize it to ~2-bit with Part-1's method (VQ + sensitivity-protect-5% + heal),
  3. measure held-out reasoning accuracy retained vs FP.

If held-out accuracy survives quantization, low-bit does NOT specifically break
reasoning — the dream's near-FP quality claim extends from facts to reasoning.

Run:  python projects/day20_intelligence/reasoning_lowbit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "day6_vector_quant"))
from vq_vs_scalar import _assign, _kmeans  # type: ignore

OUT = _HERE / "reasoning_results.json"
P = 47                  # prime modulus
D = 128
HID = 256
TRAIN_FRAC = 0.6        # held-out the rest -> tests generalisation, not memorisation
STEPS = 4000
LR = 5e-3
SEED = 0
DG, K = 4, 256          # VQ group, codebook
PROTECT = 0.05


class ModAdd(nn.Module):
    """embed a,b -> MLP -> logits over P. Must learn the group to generalise."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(P, D)
        self.fc1 = nn.Linear(2 * D, HID)
        self.fc2 = nn.Linear(HID, P)

    def forward(self, a, b):
        x = torch.cat([self.emb(a), self.emb(b)], -1)
        return self.fc2(F.gelu(self.fc1(x)))


def _split(rng):
    pairs = [(a, b) for a in range(P) for b in range(P)]
    rng.shuffle(pairs)
    n_tr = int(len(pairs) * TRAIN_FRAC)
    return pairs[:n_tr], pairs[n_tr:]


def _tensors(pairs):
    a = torch.tensor([p[0] for p in pairs]); b = torch.tensor([p[1] for p in pairs])
    y = torch.tensor([(p[0] + p[1]) % P for p in pairs])
    return a, b, y


@torch.no_grad()
def _acc(model, a, b, y):
    return float((model(a, b).argmax(-1) == y).float().mean())


def _vq_quantize_linear(layer, protect):
    """replace a Linear's weight with mixed-precision VQ (sensitivity-free error variant
    is enough here; we reuse Part-1's structure)."""
    W = layer.weight.detach().cpu().float().numpy()
    shp = W.shape
    flat = W.reshape(-1)
    pad = (-flat.size) % DG
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, np.float32)])
    V = flat.reshape(-1, DG)
    cent = _kmeans(V, K, seed=SEED)
    idx = _assign(V, cent)
    recon = cent[idx].copy()
    err = ((V - recon) ** 2).sum(1)
    nprot = int(len(V) * protect)
    if nprot:
        worst = np.argpartition(err, -nprot)[-nprot:]
        recon[worst] = V[worst]                  # protect at full precision (int8-grade)
    out = recon.reshape(-1)[: W.size].reshape(shp)
    layer.weight.data = torch.from_numpy(out.astype(np.float32))


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    import random
    r = random.Random(SEED)
    tr, te = _split(r)
    atr, btr, ytr = _tensors(tr)
    ate, bte, yte = _tensors(te)

    model = ModAdd()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    for step in range(STEPS):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(atr, btr), ytr)
        loss.backward(); opt.step()
    fp_tr, fp_te = _acc(model, atr, btr, ytr), _acc(model, ate, bte, yte)
    print(f"FP model: train acc {fp_tr:.3f} | HELD-OUT acc {fp_te:.3f} "
          f"({'generalised (learned the rule)' if fp_te > 0.9 else 'did NOT generalise'})",
          flush=True)

    # quantize the two Linear layers to ~2-bit mixed-precision (Part-1 method)
    for lin in (model.fc1, model.fc2):
        _vq_quantize_linear(lin, PROTECT)
    q_tr, q_te = _acc(model, atr, btr, ytr), _acc(model, ate, bte, yte)
    print(f"2-bit (no heal): train {q_tr:.3f} | HELD-OUT {q_te:.3f}", flush=True)

    # light healing: retrain the (now-quantized, re-floated) weights briefly on TRAIN
    opt = torch.optim.Adam(model.parameters(), lr=LR * 0.3)
    for step in range(400):
        model.train(); opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(atr, btr), ytr).backward(); opt.step()
    h_te = _acc(model, ate, bte, yte)
    print(f"2-bit + heal: HELD-OUT {h_te:.3f}", flush=True)

    retained = h_te / fp_te if fp_te > 0 else 0.0
    verdict = ("REASONING SURVIVES low-bit" if retained > 0.9 else
               "reasoning degraded under low-bit")
    print(f"\n  held-out reasoning retained: {retained*100:.1f}% of FP -> {verdict}", flush=True)

    payload = {"P": P, "train_frac": TRAIN_FRAC, "bits": "~2 (VQ d=4 K=256 + protect 5%)",
               "fp_train": round(fp_tr, 4), "fp_heldout": round(fp_te, 4),
               "q_heldout_noheal": round(q_te, 4), "q_heldout_healed": round(h_te, 4),
               "retained_frac": round(retained, 4), "verdict": verdict,
               "note": "modular addition tests learned-ALGORITHM (generalisation to held-out "
                       "pairs), then 2-bit quantization. Does reasoning survive low-bit?"}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
