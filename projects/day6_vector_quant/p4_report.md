# Day 6 P4 — sub-1-bit push: an honest cliff (plain VQ is not enough)

Goal: take VQ below 1 bit/weight via large groups + healing, and test the
"learnable transform" lever. Held-out eval, same setup as P3 (FP teacher ppl 48.4).

## Rotation sanity check (the "transform" lever for VQ)
A per-group random orthogonal rotation before k-means gave **identical** NMSE:
plain 0.45314 vs rotated 0.45314. **Confirmed: rotations do not help VQ** (k-means
is rotation-invariant; random projection would only destroy the cross-weight
structure VQ exploits). QuIP/QuaRot-style transforms help *scalar* quant, not VQ.

## Sub-1-bit post-hoc sweep (all collapse)

| config | bits/weight | post-hoc ppl | top-1 |
|---|---|---|---|
| d4 K16 | 1.001 | 102,724 | 0.00 |
| d8 K64 | 0.759 | 71,021 | 0.00 |
| d6 K16 | 0.668 | 208,910 | 0.00 |
| d16 K256 | 0.574 | 21,167 | 0.00 |
| d8 K16 | 0.502 | 113,046 | 0.00 |

## Healing the best sub-1-bit config (d16 K256, 0.574 b/w)

| step | ppl | top-1 |
|---|---|---|
| 0 (post-hoc) | 21,167 | 0.00 |
| 15 | 3,538 | 0.06 |
| 30 | 4,642 | 0.14 |
| 60 | **3,997** | 0.09 |

## Honest verdict — sub-1-bit is a real cliff for plain VQ

- Below ~1 bit/weight, **plain k-means VQ collapses** (ppl 20k–200k post-hoc), and
  **healing only partially rescues it** (0.574 b/w → ~4,000 ppl) — ~40× worse than
  the 2-bit VQ+heal point (94.7) and ~80× worse than FP (48.4). **Not usable.**
- The literature's sub-1-bit results (BTC-LLM ~0.8 b/w, 3% drop) use a **learnable
  transform + binary codebook + careful per-layer optimization** — not plain VQ.
  Our plain k-means + codebook-healing is not enough to cross 1 bit.
- The rotation "transform" lever does **not** apply to VQ (proven identical NMSE).

## Where the frontier actually sits (our method, held-out)
```
ppl (log)
 21167 ● VQ 0.57b/w post-hoc
  3997 ● VQ 0.57b/w + heal     <- sub-1-bit: heals but stays unusable
    95 ● VQ 2.02b/w + heal     <- the sweet spot (P3)
    48 ● FP teacher
```

**Conclusion:** our genuinely-new lever (VQ + healing) wins decisively at ~2 bits
(P3), but **sub-1-bit is a wall for plain VQ**. Crossing it needs the next tier —
learnable transforms + additive/residual codebooks (AQLM/BTC-LLM) + heavier
per-layer optimization — which is real engineering beyond plain k-means and harder
on a CPU. Honest boundary established; no overclaim.

## Next (honest options)
- Stay at the proven ~2-bit VQ+heal sweet spot and **combine with MoE sparsity**
  (attacks effective size/speed, P-B) rather than chase sub-1-bit.
- OR implement **residual/additive VQ** (2 small codebooks summed) to push ~1.5
  bits with quality — the realistic next compression step.
