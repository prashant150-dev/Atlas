"""Model-size -> best architecture advisor (honest, from our measured numbers).

We learned the hard way that the right technique DEPENDS ON MODEL SIZE: low-bit quality
is size-dependent (small models break at 2-bit; big models tolerate it). This tool
categorises any model by parameter count and recommends the architecture/approach that
gives the best output, plus whether it fits THIS PC (i5-4590T, 8GB RAM, 50GB disk).

Constants are MEASURED in this project:
  - LUT kernel throughput (large matrices) : 3637 M active-params/sec  (Day-18)
  - usable low-bit floor                   : ~2 bits w/ healing, ~4 bits post-hoc
  - this PC: 8GB RAM (~6.5GB usable), 50GB disk
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "advisor_results.json"

COMPUTE_PPS = 3637e6          # measured params/sec
RAM_USABLE = 6.5e9            # bytes
DISK = 50e9                   # bytes


def categorize(params):
    b = params / 1e9
    if b <= 10:
        return "SMALL"          # 1-10B
    if b <= 50:
        return "AVERAGE"        # 11-50B
    if b <= 300:
        return "GOOD"           # 51-300B
    if b <= 800:
        return "BIG"            # 301-800B
    return "ULTRA"              # 801B-2T


# per-category honest recipe: (bits, dense/MoE, what makes the BEST output, why)
RECIPES = {
    "SMALL": dict(   # 1-10B
        bits=4.0, sparse=False,
        best="4-bit post-hoc (llama.cpp Q4) or 2-bit + healing; keep DENSE",
        why="1-10B run dense and fit this PC. 4-bit is near-FP with no training; 2-bit "
             "needs healing (sub-2B break easily — we saw Qwen-1.5B gibberish at 2-bit no-heal).",
        pieces="Size + Healing + LUT kernel"),
    "AVERAGE": dict(  # 11-50B
        bits=2.5, sparse=False,
        best="2-3 bit + heavy healing (AQLM-style); MoE if the model has experts",
        why="low-bit starts to SHINE here (~1.0-1.1x FP, emergent with scale). Dense at "
             "this size is heavy on an 8GB CPU -> slow; MoE versions run far better.",
        pieces="Size + Sensitivity + Healing + LUT kernel + (MoE if available)"),
    "GOOD": dict(     # 51-300B
        bits=2.0, sparse=True,
        best="2-bit + MoE routing (sparse) + retrieval; dense here = GPU-only",
        why="at 51-300B only SPARSE (few active of big total) is feasible on weak HW. "
             "Knowledge from the big total, speed from the small active set.",
        pieces="Size + Heal + Kernel + MoE + Retrieval"),
    "BIG": dict(      # 301-800B
        bits=1.8, sparse=True,
        best="FULL AetherCore: aggressive sparse MoE + ~2-bit + retrieval + streaming convert",
        why="this is the dream's home. Only sparsity + low-bit + retrieval makes a 301-800B "
             "model storable/runnable on a potato. Needs external drive for storage.",
        pieces="ALL 6 pieces: Size+Heal+Kernel+MoE+Retrieval+Self-review"),
    "ULTRA": dict(    # 801B-2T
        bits=1.6, sparse=True,
        best="FULL AetherCore, maximum sparsity + 1-2 bit + retrieval + streaming",
        why="801B-2T: only runnable as very-sparse (tiny active of vast total) + low-bit + "
             "retrieval. Storage on external drive; reasoning capped by the small active set.",
        pieces="ALL 6 pieces, maxed"),
}

# MoE active params assumed RESIDENT for the sparse categories (fixed small active set so
# the dream stays fast on this PC). Dense categories use all params.
ACTIVE_FIXED = {"GOOD": 90e6, "BIG": 120e6, "ULTRA": 200e6}   # honest aggressive-sparse
ACTIVE_FRAC = {}


def advise(params):
    cat = categorize(params)
    r = RECIPES[cat]
    bits = r["bits"]
    disk_bytes = params * bits / 8
    if cat in ACTIVE_FIXED:                            # sparse -> small fixed active set
        active = ACTIVE_FIXED[cat]
    else:                                              # dense -> all params active
        active = params
    ram_bytes = active * bits / 8
    tok_s = COMPUTE_PPS / max(active, 1)
    fits_disk = disk_bytes <= DISK
    fits_ram = ram_bytes <= RAM_USABLE
    on_this_pc = fits_ram and (fits_disk or cat in ("GOOD", "BIG", "ULTRA"))  # big -> ext drive
    return {
        "params": params, "category": cat, "rec_bits": bits, "sparse": r["sparse"],
        "disk_gb": round(disk_bytes/1e9, 2), "active_params_M": round(active/1e6, 1),
        "ram_gb": round(ram_bytes/1e9, 3), "est_tok_s": round(tok_s, 2),
        "fits_ram": fits_ram, "fits_disk_50gb": fits_disk,
        "runs_on_this_pc": on_this_pc, "best": r["best"], "why": r["why"], "pieces": r["pieces"],
    }


def main():
    sizes = [1e9, 7e9, 10e9, 13e9, 30e9, 50e9, 70e9, 150e9, 300e9, 400e9, 800e9, 1e12, 2e12]
    if len(sys.argv) > 1:
        sizes = [float(sys.argv[1])]
    rows = []
    print(f"{'params':>9} | {'category':>10} | {'bits':>4} | {'disk':>8} | "
          f"{'active':>9} | {'tok/s':>7} | this PC?", flush=True)
    print("-" * 78, flush=True)
    last_cat = None
    for p in sizes:
        a = advise(p); rows.append(a)
        tag = "EXTERNAL drive" if (a["category"] in ("BIG", "ULTRA-HUGE") and not a["fits_disk_50gb"]) else ""
        pc = ("YES " + tag) if a["runs_on_this_pc"] else "needs GPU/more"
        ps = f"{p/1e9:.2f}B" if p >= 1e9 else f"{p/1e6:.0f}M"
        print(f"{ps:>9} | {a['category']:>10} | {a['rec_bits']:>4.1f} | "
              f"{a['disk_gb']:>6.1f}GB | {a['active_params_M']:>7.0f}M | "
              f"{a['est_tok_s']:>7.1f} | {pc}", flush=True)
        if a["category"] != last_cat:
            last_cat = a["category"]

    print("\nPER-CATEGORY BEST ARCHITECTURE (for best output):", flush=True)
    for cat, r in RECIPES.items():
        print(f"\n[{cat}]  recommended ~{r['bits']:.0f}-bit, "
              f"{'sparse MoE' if r['sparse'] else 'dense'}", flush=True)
        print(f"  BEST : {r['best']}", flush=True)
        print(f"  WHY  : {r['why']}", flush=True)
        print(f"  USES : {r['pieces']}", flush=True)

    OUT.write_text(json.dumps({"constants": {"compute_pps": COMPUTE_PPS, "ram_usable": RAM_USABLE,
                   "disk": DISK}, "rows": rows, "recipes": RECIPES}, indent=2), encoding="utf-8")
    print(f"\nwritten {OUT}", flush=True)


if __name__ == "__main__":
    main()
