"""Day-15: do experts SPECIALIZE by task? -> task-conditional expert loading.

The user's key insight: to run a huge model on tiny RAM, don't load the whole model
— load only the experts the current task needs (coding task -> coding experts; the
rest stay on disk). This works ONLY IF experts specialize per task/domain. We measure
that here.

Task: 4 domains, each with its OWN key->value mapping. Input = [domain_tag, key],
predict the domain-specific value. Train an 8-expert top-2 MoE; then measure, per
domain, WHICH experts the router uses. If each domain concentrates on a few distinct
experts, we can load just those (RAM saved) — proving conditional expert loading.

Run:  python projects/day15_expert_routing/conditional_experts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "day7_vq_moe"))

import vq_moe  # noqa: E402
from src.architecture.experiment import _train  # noqa: E402

D = 4            # domains
V = 16           # values/keys
SEED = 0
STEPS = 1500
BATCH = 128
LR = 3e-3
N_EXPERT = 8
TOP_K = 2
EXP_H = 64
# tokens: 0..D-1 = domain tags; D..D+V-1 = key/value symbols
VOCAB = D + V
SEQ = 3          # [domain, key, value]
OUT = _HERE / "results.json"


def make_data(n, gen):
    perms = torch.stack([torch.randperm(V, generator=gen) for _ in range(D)])  # per-domain map
    dom = torch.randint(0, D, (n,), generator=gen)
    key = torch.randint(0, V, (n,), generator=gen)
    val = perms[dom, key]
    X = torch.zeros(n, SEQ, dtype=torch.long)
    X[:, 0] = dom
    X[:, 1] = key + D
    X[:, 2] = val + D
    Y = torch.full((n, SEQ), -100, dtype=torch.long)
    Y[:, 1] = val + D      # at key position predict domain-specific value
    return X, Y, dom


def main():
    torch.manual_seed(SEED)
    vq_moe.VOCAB = VOCAB
    vq_moe.SEQ = SEQ
    from vq_moe import LM, MoEFFN

    gen = torch.Generator().manual_seed(SEED)
    X, Y, _ = make_data(9000, gen)
    lm = LM(MoEFFN(N_EXPERT, EXP_H, TOP_K))
    _train(lm, X, Y, STEPS, LR, BATCH, seed=0)

    # eval accuracy
    from src.architecture.experiment import _evaluate
    acc, _ = _evaluate(lm, X, Y)

    # per-domain routing: for each domain, which experts get used at the key position?
    geval = torch.Generator().manual_seed(SEED + 1)
    Xe, Ye, dome = make_data(4000, geval)
    with torch.no_grad():
        h = lm.embed(Xe) + lm.pos[:, :SEQ]
        h = h + lm.attn(lm.n1(h))
        fi = lm.n2(h)[:, 1, :]                 # key-position hidden
        topi = lm.ffn.router(fi).topk(TOP_K, -1).indices   # [n, top_k]

    dom_expert = np.zeros((D, N_EXPERT))
    for d in range(D):
        m = (dome == d).numpy()
        cnt = np.bincount(topi[m].reshape(-1).numpy(), minlength=N_EXPERT)
        dom_expert[d] = cnt / cnt.sum()

    # specialization metrics
    print(f"MoE accuracy: {acc:.3f}  ({D} domains, {N_EXPERT} experts, top-{TOP_K})\n", flush=True)
    print("per-domain expert usage (fraction):", flush=True)
    for d in range(D):
        top = np.argsort(dom_expert[d])[::-1]
        used = [(int(e), round(float(dom_expert[d][e]), 2)) for e in top if dom_expert[d][e] > 0.05]
        print(f"  domain {d}: {used}", flush=True)

    # how many experts cover 90% of each domain's routing?
    experts_needed = []
    for d in range(D):
        s = np.sort(dom_expert[d])[::-1].cumsum()
        experts_needed.append(int(np.searchsorted(s, 0.9) + 1))
    avg_needed = float(np.mean(experts_needed))
    # cross-domain overlap: union of top-experts across domains
    top_per_dom = [set(np.argsort(dom_expert[d])[::-1][:experts_needed[d]]) for d in range(D)]
    ram_if_conditional = avg_needed / N_EXPERT     # fraction of experts to load per task

    print(f"\nexperts needed for 90% of routing, per domain: {experts_needed} (avg {avg_needed:.1f}/{N_EXPERT})", flush=True)
    print(f"=> task-conditional loading needs ~{avg_needed:.1f}/{N_EXPERT} experts in RAM "
          f"({ram_if_conditional*100:.0f}% of MoE) -> ~{1/ram_if_conditional:.1f}x less expert-RAM", flush=True)
    specialized = avg_needed < N_EXPERT * 0.6
    print(f"specialized enough for conditional loading? {'YES' if specialized else 'PARTIAL'}", flush=True)

    OUT.write_text(json.dumps({
        "accuracy": round(acc, 4), "domains": D, "n_expert": N_EXPERT, "top_k": TOP_K,
        "per_domain_expert_usage": dom_expert.round(3).tolist(),
        "experts_needed_90pct": experts_needed, "avg_experts_needed": round(avg_needed, 2),
        "conditional_ram_fraction": round(ram_if_conditional, 3),
        "expert_ram_reduction_x": round(1/ram_if_conditional, 2),
        "specialized": bool(specialized),
    }, indent=2), encoding="utf-8")
    print(f"\nwritten {OUT}", flush=True)


if __name__ == "__main__":
    main()
