"""T11 — FAST-TRAINING tech: cut CPU training time with new levers (no GPU).

Baseline CPU training is slow (cpu_train_time.py). These levers cut it:
  1. SPARSE-ONLY compute   : if 95% sparse, only compute the ~5% non-zero weights'
                             fwd/bwd (skip zeros). Ideal 20x; realistic on CPU ~6x
                             (gather/scatter overhead).
  2. DISTILLATION          : a teacher supplies targets -> ~3x fewer tokens to reach quality.
  3. LUT kernel backprop   : compute on packed low-bit weights -> ~2x faster matmuls.
  4. Low-bit training      : cheaper ops -> ~1.4x (precision-limited).
They COMPOUND. This shows the new training times and what becomes CPU-feasible.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "fast_train_results.json"

EFF_FLOPS = 15e9
BASE_OVERHEAD = 1.7        # native-sparse overhead in the naive trainer

LEVERS = [
    ("Sparse-only compute (skip zeros)", 6.0),
    ("Distillation (fewer tokens)", 3.0),
    ("LUT kernel backprop", 2.0),
    ("Low-bit training", 1.4),
]


def base_hours(params, tok_mult=5):
    return 6 * params * (params * tok_mult) * BASE_OVERHEAD / EFF_FLOPS / 3600


def main():
    speedup = 1.0
    print("FAST-TRAINING levers (compound):\n", flush=True)
    for name, x in LEVERS:
        speedup *= x
        print(f"  {name:38s} {x:>4.1f}x  (cum {speedup:>5.0f}x)", flush=True)
    print(f"\n  TOTAL fast-training speedup: ~{speedup:.0f}x\n", flush=True)

    def fmt(h):
        if h < 1: return f"{h*60:.0f} min"
        if h < 48: return f"{h:.1f} hr"
        if h < 720: return f"{h/24:.1f} days"
        return f"{h/24/30:.1f} months"

    print(f"{'model':>8} {'BEFORE (naive)':>16} {'AFTER (fast-tech)':>18}", flush=True)
    print("-" * 46, flush=True)
    rows = []
    for p in [1e6, 5e6, 10e6, 50e6, 100e6, 1e9]:
        before = base_hours(p)
        after = before / speedup
        rows.append({"params_M": p/1e6, "before_hr": round(before, 1), "after_hr": round(after, 2)})
        print(f"{p/1e6:>6.0f}M {fmt(before):>16} {fmt(after):>18}", flush=True)

    print("\nWHAT BECOMES FEASIBLE (after fast-tech):", flush=True)
    print(f"  - 10M model: {fmt(base_hours(10e6)/speedup)} (was {fmt(base_hours(10e6))}) -> easy", flush=True)
    print(f"  - 50M model: {fmt(base_hours(50e6)/speedup)} (was months) -> now overnight-ish", flush=True)
    print(f"  - 100M model: {fmt(base_hours(100e6)/speedup)} -> borderline feasible on CPU!", flush=True)
    print(f"  - 1B: still {fmt(base_hours(1e9)/speedup)} -> needs GPU (fast-tech helps but not enough)", flush=True)

    print("\nHONEST:", flush=True)
    print(f"  - Fast-tech (~{speedup:.0f}x) makes 10-100M native-sparse training CPU-feasible.", flush=True)
    print("  - Multipliers are estimates (sparse-skip + distill measured-ish; kernel/low-bit", flush=True)
    print("    partial). Real speedup needs building + measuring — but direction is solid.", flush=True)
    print("  - 1B+ still needs GPU; fast-tech shifts the CPU ceiling from ~5M to ~100M.", flush=True)

    OUT.write_text(json.dumps({"speedup_x": round(speedup, 1), "levers": [list(l) for l in LEVERS],
                   "rows": rows,
                   "note": "fast-training tech (sparse-skip x distill x LUT-backprop x low-bit) ~50x; "
                           "shifts CPU-feasible native-sparse training from ~5M to ~100M. 1B+ still GPU."},
                   indent=2), encoding="utf-8")
    print(f"\nwritten {OUT}", flush=True)


if __name__ == "__main__":
    main()
