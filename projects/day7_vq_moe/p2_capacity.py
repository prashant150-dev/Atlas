"""Day-7 P2: a capacity-hungry task — let sparse experts show real power.

Day-7 P1 ran on char_lm, which was capacity-saturated (small ≈ big), so it only
proved bit-efficiency, not a capacity→quality gain. Here we build a task that
genuinely NEEDS capacity:

  KEYED SUBSTITUTION. There are m rules; each rule r is a fixed random permutation
  pi_r over a V-symbol alphabet. Input = [rule_r, x] and the model must output
  pi_r(x). To solve it the model must STORE all m*V mappings in its weights — a
  pure capacity demand. Small FFN bottlenecks; big FFN / MoE (experts specialise
  per rule) win.

Variants (reusing the Day-7 model classes): DenseFP-small, DenseFP-big, MoE-FP,
VQ-MoE+heal. Expect DenseFP-small < DenseFP-big (capacity matters now), and
VQ-MoE matching big at far lower stored+active bits.

Run from repo root::

    python projects/day7_vq_moe/p2_capacity.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import vq_moe  # noqa: E402  (we override its task globals before building models)
from src.architecture.experiment import _evaluate, _train  # noqa: E402

M_RULES = 80
V_SYM = 60
SEED = 0
STEPS = 3000
HEAL_STEPS = 1200
BATCH = 256
LR = 3e-3
N_EXAMPLES = 28000
SMALL_H = 16            # tighter bottleneck so capacity binds harder

# task token layout: 0 = sep (unused), rules 1..M, values M+1..M+V
VOCAB = 1 + M_RULES + V_SYM
SEQ = 3                       # [rule, x, y]
OUT = _HERE / "p2_results.json"
LOG = _HERE / "p2_log.jsonl"


def make_keyed_sub(n, gen):
    rule_off, val_off = 1, 1 + M_RULES
    perms = torch.stack([torch.randperm(V_SYM, generator=gen) for _ in range(M_RULES)])
    rule = torch.randint(0, M_RULES, (n,), generator=gen)
    x = torch.randint(0, V_SYM, (n,), generator=gen)
    y = perms[rule, x]
    inp = torch.zeros(n, SEQ, dtype=torch.long)
    inp[:, 0] = rule + rule_off
    inp[:, 1] = x + val_off
    inp[:, 2] = y + val_off
    tgt = torch.full_like(inp, -100)
    tgt[:, 1] = y + val_off       # at the x position, predict pi_rule(x)
    return inp, tgt


def _log(r):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(r) + "\n"); fh.flush()


def main():
    torch.manual_seed(SEED)
    LOG.write_text("", encoding="utf-8")
    # override the Day-7 model globals for this task
    vq_moe.VOCAB = VOCAB
    vq_moe.SEQ = SEQ
    from vq_moe import LM, DenseFFN, MoEFFN, VQMoEFFN, N_EXPERT, TOP_K, DG, K
    EXP_H = SMALL_H        # use the bottleneck hidden so capacity actually binds

    gen = torch.Generator().manual_seed(0)
    X, Y = make_keyed_sub(N_EXAMPLES, gen)
    chance = 1.0 / V_SYM
    print(f"keyed-sub: {M_RULES} rules x {V_SYM} symbols = {M_RULES*V_SYM} mappings, "
          f"vocab {VOCAB}, chance {chance:.3f}", flush=True)

    def run(label, model, steps=STEPS):
        _train(model, X, Y, steps, LR, BATCH, seed=0)
        acc, _ = _evaluate(model, X, Y)
        s, a = model.ffn.ffn_bits()
        row = {"variant": label, "acc": round(acc, 4), "ffn_stored_bits": int(s),
               "ffn_active_bits_per_token": int(a)}
        _log(row)
        print(f"  {label:16s} | acc {acc:.3f} | stored {s/1e3:7.1f}kb | active/tok {a/1e3:6.1f}kb", flush=True)
        return row, model

    t0 = time.perf_counter()
    rows = []
    print("training:", flush=True)
    r1, _ = run("DenseFP-small", LM(DenseFFN(EXP_H)))
    r2, _ = run("DenseFP-big", LM(DenseFFN(N_EXPERT * EXP_H)))
    r3, moe_model = run("MoE-FP", LM(MoEFFN(N_EXPERT, EXP_H, TOP_K)))
    rows += [r1, r2, r3]

    print("VQ-MoE (shared codebook) + healing:", flush=True)
    vq_ffn = VQMoEFFN(moe_model.ffn, DG, K, SEED)
    vq = LM(vq_ffn)
    for attr in ("embed", "attn", "n1", "n2", "head"):
        getattr(vq, attr).load_state_dict(getattr(moe_model, attr).state_dict())
    vq.pos.data.copy_(moe_model.pos.data)
    acc0, _ = _evaluate(vq, X, Y)
    print(f"  VQ-MoE post-hoc  | acc {acc0:.3f}", flush=True)
    r4, _ = run("VQ-MoE+heal", vq, steps=HEAL_STEPS)
    rows.append(r4)

    payload = {"task": "keyed_substitution", "m_rules": M_RULES, "v_sym": V_SYM,
               "chance": chance, "vq_posthoc_acc": round(acc0, 4),
               "elapsed_sec": round(time.perf_counter() - t0, 1), "variants": rows}
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
