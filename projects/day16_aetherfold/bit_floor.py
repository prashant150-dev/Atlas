"""How low can bits/weight go? Storage-arithmetic vs usable-quality floor.

Two different floors, often confused:
  1. STORAGE floor: any bits/weight is reachable via SPARSITY (most weights = 0).
     A ternary weight with zero-fraction s has entropy
        H(s) = -s*log2(s) - (1-s)*log2((1-s)/2)
     so to hit a target bits/weight you just need enough zeros.
  2. USABLE-QUALITY floor: at what bits/weight does the model still work? Empirical;
     our P4 + the literature bound this far above the storage floor.
"""
import math


def ternary_entropy(s):
    """bits/weight for ternary {-1,0,+1} with P(0)=s, P(+1)=P(-1)=(1-s)/2."""
    if s <= 0:
        return 1.0
    if s >= 1:
        return 0.0
    nz = (1 - s) / 2
    return -s * math.log2(s) - 2 * nz * math.log2(nz)


def sparsity_for_bits(target):
    """find zero-fraction s giving ternary entropy = target bits (binary search)."""
    lo, hi = 0.0, 0.999999
    for _ in range(60):
        mid = (lo + hi) / 2
        if ternary_entropy(mid) > target:
            lo = mid
        else:
            hi = mid
    return mid


P = 400e9
print("=" * 78)
print("400B model — size & feasibility at different bits/weight")
print("=" * 78)
print(f"{'bits/wt':>8} | {'400B size':>10} | {'fits 50GB?':>10} | {'ternary zeros needed':>20} | quality")
rows = []
for b in [2.04, 1.58, 1.0, 0.8, 0.58, 0.40, 0.30, 0.10]:
    gb = P * b / 8 / 1e9
    fits = "YES" if gb <= 50 else "no"
    s = sparsity_for_bits(b) if b < 1.58 else (0.333 if abs(b - 1.58) < 0.01 else 0.0)
    if b >= 1.58:
        q = "near-FP (BitNet proven)"
    elif b >= 1.0:
        q = "usable (degrades)"
    elif b >= 0.7:
        q = "bleeding-edge (BTC-LLM ~3% drop, big models + heavy method)"
    elif b >= 0.4:
        q = "research-only; our P4 collapsed (~4000 ppl)"
    else:
        q = "no usable model exists today"
    print(f"{b:>8.2f} | {gb:>8.0f} GB | {fits:>10} | {s*100:>18.1f}% | {q}")
    rows.append({"bits": b, "gb": round(gb, 1), "fits_50GB": gb <= 50,
                 "ternary_zero_fraction": round(s, 4), "quality": q})

print("\n" + "=" * 78)
print("THE TWO FLOORS")
print("=" * 78)
print("STORAGE floor : ~0 bits/weight is reachable in principle — at 0.58 bits a")
print("                ternary matrix needs ~%.0f%% zeros (extreme sparsity)." % (sparsity_for_bits(0.58)*100))
print("                400B @ 0.58-bit = %.0f GB -> FITS your 50GB disk." % (P*0.58/8/1e9))
print("USABLE floor  : measured/literature ~1.0-1.58 bits keeps capability;")
print("                ~0.8 bit is bleeding edge (heavy method, big model, small drop);")
print("                <=0.5 bit: no model retains usable quality today (our P4 confirms).")
print("CONCLUSION    : you CAN store 400B at 0.58 bits (29 GB) — but it would be a")
print("                ~95%-sparse ternary model that, with today's methods, does NOT")
print("                retain quality. Storage is not the wall; KEEPING IT SMART is.")
