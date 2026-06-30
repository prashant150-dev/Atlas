"""Day-12 Phase D: does capacity buy REASONING (generalization), preserved under
compression? — the honest intelligence probe.

Critique #2: capacity/memorization tasks aren't intelligence. So here the task
needs an ALGORITHM, tested on HELD-OUT (unseen) operand pairs — generalization,
not memorization:

  modular addition:  [a, +, b, =, c]  with c = (a+b) mod P
  train on a fraction of (a,b) pairs; TEST on UNSEEN pairs.

Test accuracy on unseen pairs = real reasoning (memorization would give high train,
chance test). We compare DenseFP-small vs DenseFP-big vs VQ-MoE+heal to ask:
  (1) does more capacity improve GENERALIZATION (not just train)?
  (2) does the compressed VQ-MoE RETAIN the reasoning?

Honest scope: tiny model, one synthetic algorithm; true "beast intelligence" needs
scale (hardware-gated). This probes the *mechanism* (capacity->reasoning, preserved
under compression), not 400B capability.

Run from repo root::

    python projects/day12_intelligence/phase_d_reasoning.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "day7_vq_moe"))

import vq_moe  # noqa: E402
from src.architecture.experiment import _evaluate, _train  # noqa: E402

OUT = _HERE / "phase_d_results.json"
LOG = _HERE / "phase_d_log.jsonl"
P = 31                       # modulus
PLUS, EQ = P, P + 1
VOCAB = P + 2
SEQ = 5                      # [a, +, b, =, c]
TRAIN_FRAC = 0.80
STEPS = 2500
HEAL_STEPS = 800
BATCH = 128
LR = 3e-3
SEED = 0


def make_split():
    g = torch.Generator().manual_seed(SEED)
    pairs = [(a, b) for a in range(P) for b in range(P)]
    perm = torch.randperm(len(pairs), generator=g).tolist()
    pairs = [pairs[i] for i in perm]
    n_tr = int(len(pairs) * TRAIN_FRAC)
    tr, te = pairs[:n_tr], pairs[n_tr:]

    def to_tensors(ps):
        X = torch.zeros(len(ps), SEQ, dtype=torch.long)
        Y = torch.full((len(ps), SEQ), -100, dtype=torch.long)
        for i, (a, b) in enumerate(ps):
            c = (a + b) % P
            X[i] = torch.tensor([a, PLUS, b, EQ, c])
            Y[i, 3] = c            # at the '=' position, predict c
        return X, Y
    return to_tensors(tr), to_tensors(te), len(tr), len(te)


def _log(r):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(r) + "\n"); fh.flush()


def main():
    torch.manual_seed(SEED)
    vq_moe.VOCAB = VOCAB
    vq_moe.SEQ = SEQ
    from vq_moe import LM, DenseFFN, MoEFFN, VQMoEFFN, N_EXPERT, TOP_K, DG, K
    EXP_H = 64

    (Xtr, Ytr), (Xte, Yte), n_tr, n_te = make_split()
    print(f"modular add mod {P}: train {n_tr} pairs, test {n_te} UNSEEN pairs (chance {1/P:.3f})", flush=True)
    rows = []

    def run(label, model, steps=STEPS):
        _train(model, Xtr, Ytr, steps, LR, BATCH, seed=0)
        tr_acc, _ = _evaluate(model, Xtr, Ytr)
        te_acc, _ = _evaluate(model, Xte, Yte)
        row = {"variant": label, "train_acc": round(tr_acc, 4), "test_acc": round(te_acc, 4),
               "gap": round(tr_acc - te_acc, 4)}
        rows.append(row); _log(row)
        print(f"  {label:16s} | train {tr_acc:.3f} | TEST(unseen) {te_acc:.3f} | gap {row['gap']:+.3f}", flush=True)
        return model

    t0 = time.perf_counter()
    print("training (generalization = TEST on unseen pairs):", flush=True)
    run("DenseFP-small", LM(DenseFFN(EXP_H)))
    run("DenseFP-big", LM(DenseFFN(N_EXPERT * EXP_H)))
    moe = run("MoE-FP", LM(MoEFFN(N_EXPERT, EXP_H, TOP_K)))

    print("VQ-MoE + heal (compressed — does it retain reasoning?):", flush=True)
    vq = LM(VQMoEFFN(moe.ffn, DG, K, SEED))
    for attr in ("embed", "attn", "n1", "n2", "head"):
        getattr(vq, attr).load_state_dict(getattr(moe, attr).state_dict())
    vq.pos.data.copy_(moe.pos.data)
    run("VQ-MoE+heal", vq, steps=HEAL_STEPS)

    payload = {"task": f"modular_addition_mod_{P}", "chance": 1 / P,
               "train_pairs": n_tr, "test_pairs_unseen": n_te,
               "elapsed_sec": round(time.perf_counter() - t0, 1), "variants": rows}
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
