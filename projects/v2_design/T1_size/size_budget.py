"""T1 SIZE — the 101x decomposition: how to reach 101x smaller, lever by lever.

fp16 = 16 bits/weight. 101x smaller = 0.158 bits/weight. But Day-1 proved you CANNOT push
a single weight below ~1-2 bits without quality collapse (information floor). So 101x can
NOT come from squeezing each weight — it must come from a STACK of structural levers, each
removing redundancy a different way. This tool multiplies the levers and shows the honest
path + the quality risk of each.

Run:  python projects/v2_design/T1_size/size_budget.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "size_budget_results.json"
FP16_BITS = 16.0
TARGET_X = 101.0

# Each lever: (name, multiplier, status, quality_risk, how). Multipliers COMPOUND.
LEVERS = [
    ("Low-bit VQ (2-bit)", 8.0, "DONE",
     "low (with healing ~1.4x FP)",
     "16-bit -> 2-bit via mixed-precision vector quantization (our Day-17)"),
    ("Sparsity / pruning", 4.0, "TODO",
     "medium (must prune only unused weights; healing recovers)",
     "store only non-zero weights; ~75% pruned = 4x fewer to store"),
    ("Cross-layer shared codebook", 1.5, "PROVEN-small",
     "tiny (1.017x penalty, Day-16)",
     "one codebook for all layers instead of per-layer; grows with depth"),
    ("Embedding compression", 1.6, "TODO",
     "medium (embeddings are sensitive; factorize/low-rank not 2-bit)",
     "huge vocab embedding table -> low-rank factorization + share"),
    ("Structure / dedup (delta, low-rank)", 1.4, "PARTIAL",
     "medium (only where real redundancy exists)",
     "low-rank residuals + dedup repeated blocks"),
]


def main():
    print(f"baseline: fp16 = {FP16_BITS} bits/weight", flush=True)
    print(f"101x target = {FP16_BITS/TARGET_X:.3f} bits/weight (effective)\n", flush=True)
    print(f"{'lever':38s} {'x':>5} {'cumulative':>11} {'eff bits/wt':>12}  status", flush=True)
    print("-" * 88, flush=True)
    cum = 1.0
    rows = []
    for name, mult, status, risk, how in LEVERS:
        cum *= mult
        eff_bits = FP16_BITS / cum
        rows.append({"lever": name, "x": mult, "cumulative_x": round(cum, 1),
                     "eff_bits_per_weight": round(eff_bits, 3), "status": status,
                     "quality_risk": risk, "how": how})
        print(f"{name:38s} {mult:>4.1f}x {cum:>9.1f}x {eff_bits:>11.3f}   {status}", flush=True)

    print("-" * 88, flush=True)
    print(f"\nTOTAL stacked: {cum:.1f}x  (target {TARGET_X:.0f}x)", flush=True)
    if cum >= TARGET_X:
        print(f"-> 101x is REACHABLE by stacking these levers (at the listed quality risks).", flush=True)
    else:
        print(f"-> stack reaches {cum:.0f}x; need {TARGET_X/cum:.1f}x more from a further lever.", flush=True)

    print("\nHONEST NOTES:", flush=True)
    print("- 101x is NOT one trick — it is ~5 levers compounding. Each costs SOME quality;", flush=True)
    print("  the job is to stack them while HEALING keeps quality near FP.", flush=True)
    print("- Pure per-weight bits hit a ~1-2 bit floor (Day-1). Past 8x, gains come from", flush=True)
    print("  STRUCTURE (sparsity, sharing, factorization), not smaller numbers.", flush=True)
    print("- Where we are: lever 1 DONE (8x, ~1.4x FP). Levers 2-5 are the T1 roadmap.", flush=True)

    payload = {"fp16_bits": FP16_BITS, "target_x": TARGET_X,
               "target_bits_per_weight": round(FP16_BITS/TARGET_X, 3),
               "levers": rows, "total_stacked_x": round(cum, 1),
               "reachable": cum >= TARGET_X,
               "note": "101x size = stack of ~5 structural levers compounding, NOT one trick; "
                       "past the ~1-2 bit/weight floor gains come from structure + healing."}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten {OUT}", flush=True)


if __name__ == "__main__":
    main()
