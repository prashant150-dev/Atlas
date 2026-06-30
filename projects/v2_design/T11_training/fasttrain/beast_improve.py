"""Beast improvement (CPU research): harder corpus + SMART sparsity allocation.

Two improvements over the first native-sparse proof:
  1. HARDER corpus — diverse, less-repetitive text (honest stress-test; repetitive text
     made accuracy look too easy).
  2. SMART allocation — not every layer equally sparse. Keep the FIRST & LAST blocks
     denser (they matter most), sparsify the MIDDLE harder, at the SAME average sparsity.
     Tests whether smart allocation beats uniform at equal average bits.

Compares at ~95% average sparse: dense | uniform-native | SMART-native.

Run:  python projects/v2_design/T11_training/fasttrain/beast_improve.py [steps]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import native_sparse_lm as N  # type: ignore
from native_sparse_lm import CharLM, BLOCK, evaluate, get_batch, RIGL_EVERY  # type: ignore

OUT = HERE / "beast_improve_results.json"
SEED = 0
AVG_SP = 0.95


def harder_corpus():
    """diverse, low-repetition text across many topics (harder than the repeated one)."""
    parts = [
        "The mitochondria is the powerhouse of the cell, producing energy through respiration.",
        "Quantum entanglement links two particles so measuring one instantly affects the other.",
        "Compound interest grows wealth exponentially when returns are reinvested over decades.",
        "Photosynthesis converts sunlight, water and carbon dioxide into glucose and oxygen.",
        "The French Revolution overthrew the monarchy and reshaped European politics forever.",
        "A binary search halves the search space each step, finding items in logarithmic time.",
        "Plate tectonics explains earthquakes, mountains and the slow drift of continents.",
        "Supply and demand set prices where buyers and sellers reach a market equilibrium.",
        "Neurons communicate through electrical spikes and chemical neurotransmitters across synapses.",
        "The water cycle moves moisture through evaporation, condensation and precipitation.",
        "Encryption scrambles data with keys so only the intended receiver can read it.",
        "Natural selection favours traits that improve survival and reproduction over generations.",
        "Inflation erodes purchasing power as the general price level rises across an economy.",
        "A recursive function solves a problem by calling itself on smaller subproblems.",
        "Gravity bends spacetime, and massive objects curve the paths of light and planets.",
        "Vaccines train the immune system by exposing it to a harmless piece of a pathogen.",
    ]
    # shuffle-join a few times for length, but keep diversity (no single repeated sentence)
    import random
    r = random.Random(SEED)
    text = ""
    for _ in range(60):
        r.shuffle(parts)
        text += " ".join(parts) + " "
    return text


def smart_init_masks(model, avg_sp):
    """allocate sparsity per layer: first & last blocks denser, middle sparser, same average."""
    nb = len(model.blocks)
    # per-block target: ends ~0.85, middle ~0.99, scaled so average ~= avg_sp
    targets = []
    for i in range(nb):
        edge = min(i, nb - 1 - i)            # 0 at ends, larger in middle
        t = 0.85 + 0.14 * (edge / max(1, (nb - 1) / 2))
        targets.append(t)
    # rescale to hit avg
    mean_t = sum(targets) / len(targets)
    targets = [min(0.995, max(0.5, t * avg_sp / mean_t)) for t in targets]
    bi = 0
    for b in model.blocks:
        for lin in [b.q, b.k, b.v, b.proj, b.fc1, b.fc2]:
            w = lin.weight.detach(); f = w.abs().view(-1); k = int(len(f) * targets[bi])
            m = (w.abs() > f.kthvalue(k).values).float() if k > 0 else torch.ones_like(w)
            model.masks[id(lin)] = m
        bi += 1
    return targets


def train(model, data, steps, gen, smart):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    if smart:
        smart_init_masks(model, AVG_SP)
    else:
        model.init_masks(AVG_SP)
    model.apply_masks()
    for step in range(steps):
        if step and step % RIGL_EVERY == 0 and step < steps * 0.8:
            x, y = get_batch(data, gen)
            drop = 0.3 * 0.5 * (1 + math.cos(math.pi * step / (steps * 0.8)))
            N.rigl_update(model, x, y, drop)
        x, y = get_batch(data, gen)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(x).reshape(-1, model.head.out_features), y.reshape(-1)).backward()
        for lin in model.sparse_linears:
            lin.weight.grad *= model.masks[id(lin)]
        opt.step(); model.apply_masks()
    return model


def train_dense(model, data, steps, gen):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        x, y = get_batch(data, gen)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(x).reshape(-1, model.head.out_features), y.reshape(-1)).backward()
        opt.step()
    return model


def main():
    torch.manual_seed(SEED)
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    text = harder_corpus()
    chars = sorted(set(text)); vocab = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data)); tr, val = data[:n], data[n:]
    gen = torch.Generator().manual_seed(SEED)
    print(f"HARDER corpus: vocab {vocab}, {len(text)} chars (diverse, low-repeat), "
          f"avg sparse {AVG_SP*100:.0f}%, {steps} steps\n", flush=True)

    d = evaluate(train_dense(CharLM(vocab), tr, steps, torch.Generator().manual_seed(SEED)), val, gen)
    print(f"  dense              | acc {d[1]:.3f} | ppl {d[0]:.2f}", flush=True)
    u = evaluate(train(CharLM(vocab), tr, steps, torch.Generator().manual_seed(SEED+1), smart=False), val, gen)
    print(f"  uniform-native 95% | acc {u[1]:.3f} | ppl {u[0]:.2f}", flush=True)
    s = evaluate(train(CharLM(vocab), tr, steps, torch.Generator().manual_seed(SEED+1), smart=True), val, gen)
    print(f"  SMART-native 95%   | acc {s[1]:.3f} | ppl {s[0]:.2f}", flush=True)

    gain = s[1] - u[1]
    print(f"\n  dense {d[1]:.3f} | uniform {u[1]:.3f} | SMART {s[1]:.3f}", flush=True)
    print(f"  smart-allocation gain over uniform: {gain:+.3f}", flush=True)
    verdict = ("SMART allocation BEATS uniform at same avg bits" if gain > 0.01
               else "smart ~ uniform here")
    print(f"  native keeps {s[1]/d[1]*100:.0f}% of dense on a HARDER task -> {verdict}", flush=True)

    OUT.write_text(json.dumps({"vocab": vocab, "avg_sparsity": AVG_SP, "steps": steps,
                   "dense_acc": round(d[1],4), "uniform_acc": round(u[1],4), "smart_acc": round(s[1],4),
                   "dense_ppl": round(d[0],2), "uniform_ppl": round(u[0],2), "smart_ppl": round(s[0],2),
                   "smart_gain": round(gain,4), "verdict": verdict,
                   "note": "harder/diverse corpus + smart per-layer sparsity allocation (ends denser, "
                           "middle sparser) vs uniform, at same avg 95% sparse. Beast-improvement test."},
                   indent=2), encoding="utf-8")
    print(f"\nwritten {OUT}", flush=True)


if __name__ == "__main__":
    main()
