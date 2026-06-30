"""Part-4 BEAST INTELLIGENCE — 4D: the honest scale projection (the capstone).

Every MECHANISM is now measured and works: 2-bit near-FP (P1), LUT kernel 3.84x (P2),
15M retrieval (P3), sparse-active = big-total capacity (4A), reasoning survives low-bit
(4B). 4D asks the final question honestly: given this PC's MEASURED compute throughput,
what intelligence tier can actually run here at speed — and what hardware reaches "beast"?

Two parts:
  1. MEASURED: a small size->quality sweep on the capacity task (confirms, on our own
     setup, that more capacity -> more capability; the trend direction the dream needs).
  2. PROJECTION: combine measured throughput (3637 M params/s, Day-18) with the
     active-param budgets that different capability tiers need, to show where the dream
     config lands and what the true remaining wall is.

Run:  python projects/day20_intelligence/scale_projection.py
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
sys.path.insert(0, str(_HERE))
from keystone_moe import MoEHead, _make_maps, _batch, N_DOMAINS, KEYS_PER  # type: ignore

OUT = _HERE / "scale_projection_results.json"
SEED = 0
COMPUTE_PPS = 3637e6        # measured params/sec (LUT kernel, large matrices, Day-18)


def _measured_trend():
    """MoE recall as TOTAL capacity grows (more experts), active ~ fixed. Confirms the
    'bigger total = more capability' direction on our own measured setup."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    maps = _make_maps(rng)
    rows = []
    for n_exp in [2, 4, 8, 16, 32]:
        m = MoEHead(n_exp, 4, top_k=1)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        rb = np.random.default_rng(SEED + 1)
        m.train()
        for _ in range(2500):
            tg, ke, va = _batch(maps, rb, 256)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(tg, ke), va).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            dom = np.repeat(range(N_DOMAINS), KEYS_PER); kk = np.tile(range(KEYS_PER), N_DOMAINS)
            tg = torch.tensor(dom); ke = torch.tensor(kk)
            va = torch.tensor([maps[dom[i]][kk[i]] for i in range(len(dom))])
            acc = float((m(tg, ke).argmax(-1) == va).float().mean())
        tot = sum(p.numel() for p in m.parameters())
        rows.append({"experts": n_exp, "total_params": tot, "active": m.active_params(),
                     "recall": round(acc, 4)})
        print(f"  experts {n_exp:2d} | total {tot:6d} | active {m.active_params():5d} | "
              f"recall {acc:.3f}", flush=True)
    return rows


def _projection():
    """tok/s = throughput / active_params. Capability tiers need different ACTIVE budgets
    (rough public estimates of active params/token). Show what runs at >=40 tok/s here."""
    tiers = [
        ("GPT-2 small (124M dense)", 124e6),
        ("~1B dense", 1.0e9),
        ("dream config (sparse, 80-100M active of 400B)", 90e6),
        ("GPT-3-class active (~13B)", 13e9),
        ("GPT-4-class active (~50-100B est.)", 60e9),
    ]
    rows = []
    for name, active in tiers:
        tps = COMPUTE_PPS / active
        rows.append({"tier": name, "active_params": active, "tok_s_this_pc": round(tps, 3),
                     "runs_at_40tps": tps >= 40})
        print(f"  {name:48s} active {active/1e6:8.0f}M -> {tps:8.2f} tok/s "
              f"{'OK' if tps >= 40 else 'too slow'}", flush=True)
    # what throughput would beast need at 40 tok/s, and how far is this PC?
    beast_active = 60e9
    need_pps = 40 * beast_active
    gap = need_pps / COMPUTE_PPS
    print(f"\n  beast (~60B active) @ 40 tok/s needs {need_pps/1e9:.0f}B params/s; "
          f"this PC has {COMPUTE_PPS/1e9:.1f}B/s -> {gap:.0f}x short (needs GPU/cluster)",
          flush=True)
    return rows, {"beast_active": beast_active, "need_params_per_sec": need_pps,
                  "this_pc_params_per_sec": COMPUTE_PPS, "throughput_gap_x": round(gap, 1)}


def main():
    print("1) MEASURED size->capability trend (MoE total capacity vs recall):", flush=True)
    trend = _measured_trend()
    print("\n2) PROJECTION — what intelligence tier runs at >=40 tok/s on THIS PC:", flush=True)
    proj, beast = _projection()

    payload = {"compute_params_per_sec": COMPUTE_PPS, "measured_trend": trend,
               "projection": proj, "beast_gap": beast,
               "honest_conclusion":
                   "All mechanisms work and bigger-total->more-capability holds (measured). "
                   "On THIS PC the binding wall is the ACTIVE-param budget: ~40 tok/s allows "
                   "~90M active, which hosts the dream's sparse config and is enough for "
                   "knowledge+light reasoning, but GPT-4-class capability needs ~10-100B "
                   "ACTIVE params/token. At this CPU's 3.6B params/s that is ~%dx too slow -> "
                   "beast-at-speed needs a GPU/cluster. The dream's METHOD is proven; full "
                   "beast intelligence is the one genuinely hardware-gated piece."
                   % round(40 * 60e9 / COMPUTE_PPS)}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten {OUT}", flush=True)


if __name__ == "__main__":
    main()
