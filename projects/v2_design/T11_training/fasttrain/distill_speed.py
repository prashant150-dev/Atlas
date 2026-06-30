"""Fast-CPU-training research #2: does DISTILLATION reach quality in FEWER steps?

Per-step speed (sparse-skip) is size-limited on CPU. The orthogonal lever is FEWER STEPS:
a teacher's soft targets carry more signal than hard labels, so the student converges
faster. We train a native-sparse char-LM student two ways and measure accuracy at step
checkpoints:
  scratch  : cross-entropy on data (hard labels)
  distill  : KL from a dense TEACHER's soft logits (+ a little CE)
If distill hits the same accuracy in fewer steps, it's a real fast-training multiplier.

Run:  python projects/v2_design/T11_training/fasttrain/distill_speed.py [max_steps]
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
from native_sparse_lm import (BATCH, BLOCK, CharLM, SPARSITY, _corpus, evaluate,  # type: ignore
                              get_batch, rigl_update, RIGL_EVERY)

OUT = HERE / "distill_speed_results.json"
SEED = 0


def train_track(model, teacher, data, val, max_steps, ckpts, gen, distill):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.init_masks(SPARSITY); model.apply_masks()
    track = {}
    for step in range(max_steps + 1):
        if step in ckpts:
            track[step] = round(evaluate(model, val, gen)[1], 4)
        if step == max_steps:
            break
        if step and step % RIGL_EVERY == 0 and step < max_steps * 0.8:
            x, y = get_batch(data, gen)
            drop = 0.3 * 0.5 * (1 + math.cos(math.pi * step / (max_steps * 0.8)))
            rigl_update(model, x, y, drop)
        x, y = get_batch(data, gen)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        if distill:
            with torch.inference_mode():
                tl = teacher(x).detach()
            T = 2.0
            kl = F.kl_div(F.log_softmax(logits / T, -1), F.softmax(tl / T, -1),
                          reduction="batchmean") * T * T
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss = 0.9 * kl + 0.1 * ce
        else:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        for lin in model.sparse_linears:
            lin.weight.grad *= model.masks[id(lin)]
        opt.step(); model.apply_masks()
    return track


def main():
    torch.manual_seed(SEED)
    max_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 900
    text = _corpus(); chars = sorted(set(text)); vocab = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data)); train_d, val_d = data[:n], data[n:]
    gen = torch.Generator().manual_seed(SEED)
    ckpts = {0, 150, 300, 500, 700, max_steps}

    # teacher = a dense model trained briefly
    print("training dense teacher...", flush=True)
    teacher = CharLM(vocab)
    topt = torch.optim.Adam(teacher.parameters(), lr=3e-3)
    for _ in range(900):
        x, y = get_batch(train_d, gen)
        topt.zero_grad(set_to_none=True)
        F.cross_entropy(teacher(x).reshape(-1, vocab), y.reshape(-1)).backward(); topt.step()
    teacher.eval()
    print(f"teacher acc {evaluate(teacher, val_d, gen)[1]:.3f}\n", flush=True)

    print("training native-sparse student: SCRATCH vs DISTILL\n", flush=True)
    g1 = torch.Generator().manual_seed(SEED + 1)
    scratch = train_track(CharLM(vocab), teacher, train_d, val_d, max_steps, ckpts, g1, distill=False)
    g2 = torch.Generator().manual_seed(SEED + 1)
    distill = train_track(CharLM(vocab), teacher, train_d, val_d, max_steps, ckpts, g2, distill=True)

    print(f"{'step':>6} {'scratch acc':>12} {'distill acc':>12}", flush=True)
    print("-" * 32, flush=True)
    for s in sorted(ckpts):
        print(f"{s:>6} {scratch.get(s, 0):>12.3f} {distill.get(s, 0):>12.3f}", flush=True)

    # steps to reach 90% of final scratch accuracy
    target = 0.9 * scratch[max_steps]
    def steps_to(track):
        for s in sorted(track):
            if track[s] >= target:
                return s
        return max_steps
    ss, ds = steps_to(scratch), steps_to(distill)
    print(f"\n  target acc {target:.3f}: scratch @ {ss} steps | distill @ {ds} steps", flush=True)
    spd = ss / max(ds, 1)
    print(f"  -> distillation reaches it {spd:.1f}x FASTER (fewer steps = less CPU time)"
          if spd > 1.1 else "  -> distill ~ scratch here", flush=True)

    OUT.write_text(json.dumps({"max_steps": max_steps, "teacher_acc": round(evaluate(teacher, val_d, gen)[1], 4),
                   "scratch": scratch, "distill": distill, "steps_scratch": ss, "steps_distill": ds,
                   "speedup": round(spd, 2),
                   "note": "distillation vs scratch for a native-sparse char-LM; fewer steps to "
                           "quality = a real fast-training lever (orthogonal to per-step speed)."},
                   indent=2), encoding="utf-8")
    print(f"\nwritten {OUT}", flush=True)


if __name__ == "__main__":
    main()
