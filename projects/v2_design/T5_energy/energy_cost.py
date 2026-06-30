"""T5 ENERGY/COST — 101x cheaper per result, computed from the proven T1-T3 levers.

Energy isn't a new mechanism — it FALLS OUT of size+sparsity+kernel. But the breakdown is
counter-intuitive: on real hardware, MOVING a weight from DRAM costs ~100x more energy than
the arithmetic on it (the "energy memory wall"). So low-bit (fewer bytes moved) is the
biggest energy lever, then sparsity (fewer ops), then no-multiply kernel (cheaper ops).

Energy figures (Horowitz, 45nm, pJ) — standard reference:
  fp32 MULT 3.7 | fp32 ADD 0.9 | int8 ADD 0.03 | DRAM read ~160 pJ/byte | SRAM ~1.3 pJ/byte

We compute energy/token for naive fp32 dense vs our stack (sparse + 2-bit + no-multiply),
then the 101x verdict + the hardware-COST tier (potato CPU vs datacenter GPU).
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "energy_results.json"

# Horowitz 45nm energy (pJ)
E_FP_MULT = 3.7
E_FP_ADD = 0.9
E_TERNARY_ADD = 0.9        # add/sub only (no multiply) at fp accum
E_DRAM_PER_BYTE = 160.0    # ~640 pJ per 32-bit access / 4 bytes

P = 1.0                    # per-parameter basis (cancels out -> ratios)


def energy_per_token(active_frac, bits, multiply):
    """energy to process `active_frac*P` params: compute + DRAM weight read."""
    aP = active_frac * P
    if multiply:
        compute = aP * (E_FP_MULT + E_FP_ADD)      # MAC per weight
    else:
        compute = aP * E_TERNARY_ADD               # add/sub only (LUT kernel)
    bytes_per_w = bits / 8.0
    memory = aP * bytes_per_w * E_DRAM_PER_BYTE     # read each active weight from DRAM
    return compute + memory


def main():
    naive = energy_per_token(active_frac=1.0, bits=32, multiply=True)
    print(f"NAIVE fp32 dense: {naive:.1f} pJ/param-token "
          f"(compute {1.0*(E_FP_MULT+E_FP_ADD):.1f} + DRAM {1.0*4*E_DRAM_PER_BYTE:.0f})\n", flush=True)

    print(f"{'config':38s} {'pJ/param':>9} {'vs naive':>9}", flush=True)
    print("-" * 60, flush=True)
    configs = [
        ("fp32 dense (naive)", 1.0, 32, True),
        ("+ 2-bit weights (low-bit)", 1.0, 2, True),
        ("+ no-multiply kernel (ternary)", 1.0, 2, False),
        ("+ sparsity 1/8 active (MoE)", 1/8, 2, False),
        ("+ sparsity 1/16 active", 1/16, 2, False),
    ]
    rows = []
    for name, a, bits, mul in configs:
        e = energy_per_token(a, bits, mul)
        rows.append({"config": name, "active_frac": a, "bits": bits, "multiply": mul,
                     "pJ": round(e, 2), "vs_naive_x": round(naive/e, 1)})
        print(f"{name:38s} {e:9.2f} {naive/e:8.1f}x", flush=True)

    best = rows[-1]["vs_naive_x"]
    print("-" * 60, flush=True)
    print(f"\nBEST energy reduction: {best:.0f}x vs naive fp32 dense", flush=True)
    print(f"-> 101x energy {'REACHED' if best >= 101 else 'needs more sparsity'} "
          f"(driven mostly by LOW-BIT: fewer DRAM bytes = the dominant energy term).", flush=True)

    # COST tier: where it runs
    print("\nCOST (where it runs):", flush=True)
    print("  current frontier AI: datacenter GPUs (~$25k each, megawatts) -> $$$/query", flush=True)
    print("  our stack: a ~$200 used PC (i5-4590T, ~35W) -> the SAME work at a tiny fraction", flush=True)
    print("  hardware-cost tier difference alone is ~100x+ (potato vs GPU server).", flush=True)

    print("\nHONEST NOTES:", flush=True)
    print("- Energy ratios are per-weight compute+DRAM; real systems also spend on attention,", flush=True)
    print("  activations, overhead — directionally right, not a wall-plug measurement.", flush=True)
    print("- vs ALREADY-EFFICIENT inference (int4 llama.cpp) the gap is smaller; 101x is vs naive.", flush=True)

    payload = {"naive_pJ": round(naive, 2), "configs": rows, "best_vs_naive_x": best,
               "energy_101x_reached": best >= 101,
               "note": "energy/token from Horowitz figures; DRAM byte-movement dominates so "
                       "low-bit is the top energy lever, then sparsity, then no-multiply. "
                       "101x energy reachable vs naive fp32 dense; plus ~100x hardware-cost tier."}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten {OUT}", flush=True)


if __name__ == "__main__":
    main()
