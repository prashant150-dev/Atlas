"""T10 LATENCY — faster response via speculative decoding + small-active + prefetch.

Throughput (T3) is tokens/sec at steady state. LATENCY is how fast you get tokens out.
Two distinct latency levers beyond T3:
  1. SMALL ACTIVE params -> less compute before the first token (MoE routes to a few experts).
  2. SPECULATIVE DECODING -> a cheap DRAFT model proposes K tokens; the big model VERIFIES
     all K in ONE pass. Accepted tokens are free -> ~K tokens per 1 big-model step.

This computes the speculative-decoding speedup as a function of draft acceptance rate and
the draft/target cost ratio (the standard model), and the first-token latency from active
params. No training, CPU.

Run:  python projects/v2_design/T10_latency/speculative.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "latency_results.json"

# measured-ish: target step cost 1.0; cheap draft ~0.1 of target; propose K tokens
DRAFT_COST = 0.1
COMPUTE_PPS = 3637e6     # params/sec (from T3)


def spec_speedup(accept, K, draft_cost=DRAFT_COST):
    """expected tokens accepted per (1 target verify + K draft) cost.
    accepted ~= (1-accept^(K+1))/(1-accept) expected run length (geometric)."""
    exp_accepted = (1 - accept ** (K + 1)) / (1 - accept) if accept < 1 else K + 1
    cost = 1.0 + K * draft_cost          # one target verify + K cheap drafts
    baseline_cost_per_token = 1.0        # vanilla: 1 target step per token
    return exp_accepted / cost * baseline_cost_per_token


def main():
    print("SPECULATIVE DECODING speedup (draft proposes K, target verifies in 1 pass):\n", flush=True)
    print(f"{'accept rate':>12} {'K=4':>7} {'K=8':>7}", flush=True)
    print("-" * 30, flush=True)
    rows = []
    for a in (0.5, 0.7, 0.8, 0.9):
        s4 = spec_speedup(a, 4); s8 = spec_speedup(a, 8)
        rows.append({"accept": a, "K4": round(s4, 2), "K8": round(s8, 2)})
        print(f"{a:12.2f} {s4:6.2f}x {s8:6.2f}x", flush=True)

    print("\nFIRST-TOKEN LATENCY from active params (compute-bound):", flush=True)
    for ap, tag in [(90e6, "dream 90M active"), (1e9, "1B dense"), (7e9, "7B dense")]:
        ms = ap / COMPUTE_PPS * 1000
        print(f"  {tag:18s} {ap/1e6:6.0f}M active -> {ms:6.0f} ms/token first-token compute", flush=True)

    print("\nHONEST READ:", flush=True)
    print("- Speculative decoding gives ~2-3x at good acceptance (0.8) — real, no quality loss", flush=True)
    print("  (target verifies, so output is identical to running the big model alone).", flush=True)
    print("- Small ACTIVE params (MoE) cut first-token latency directly (90M -> ~25ms vs 7B -> ~2s).", flush=True)
    print("- '101x lower latency' = small-active (MoE) x speculative x prefetch, vs a big dense", flush=True)
    print("  model on weak HW. Like T3, it's vs the naive heavy baseline.", flush=True)

    OUT.write_text(json.dumps({"draft_cost": DRAFT_COST, "spec_rows": rows,
                   "note": "speculative decoding ~2-3x at 0.8 acceptance (lossless, target verifies); "
                           "small-active cuts first-token latency; 101x latency = small-active x "
                           "speculative x prefetch vs big-dense-on-weak-HW."}, indent=2),
                   encoding="utf-8")
    print(f"\nwritten {OUT}", flush=True)


if __name__ == "__main__":
    main()
