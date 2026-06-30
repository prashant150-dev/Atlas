# Day 16 — AetherFold (cross-layer delta coding): tested, RED light

## The new idea
All compression so far (ours + SOTA: AQLM/QuIP#/BitNet) is PER-MATRIX. The genuinely
new lever would be CROSS-matrix: if transformer layers resemble each other, store
layer-0 fully and each later layer as a small DELTA from the previous — the delta's
entropy could be far below the per-weight floor D1 measured, opening a new floor.

## Fast probe (before building anything)
Measured on GPT-2's 12 layers, per matrix role:

| role | adjacent cosine | entropy: raw → delta | VQ nmse raw / delta |
|---|---|---|---|
| attn.c_attn | +0.000 | −0.81 → −0.32 (**+0.48 worse**) | 0.108 / 0.105 |
| attn.c_proj | +0.001 | −1.08 → −0.61 (**+0.46 worse**) | 0.108 / 0.103 |
| mlp.c_fc | +0.019 | −0.90 → −0.42 (**+0.48 worse**) | 0.103 / 0.102 |
| mlp.c_proj | +0.002 | −1.07 → −0.58 (**+0.48 worse**) | 0.112 / 0.108 |

**Average adjacent-layer cosine ≈ +0.005 (≈ 0); delta entropy is 0.48 bits HIGHER
than raw.**

## Verdict: RED — the idea does not work on GPT-2
- GPT-2's layers are **nearly orthogonal** (cosine ~0), not similar. The intuition
  "layer 5 ≈ layer 6" is **false** for trained GPT-2 weights.
- Therefore the delta (Wᵢ − Wᵢ₋₁) has *more* spread than the raw weight, so
  delta-coding costs MORE bits, not fewer. Cross-layer delta compression would
  **lose** to plain per-layer VQ.
- VQ NMSE on delta ≈ on raw (no compression gain).

## Why this is a good outcome (the discipline)
We killed a plausible-but-wrong idea in ~30 seconds of measurement instead of
building it for days. "Measure don't assume" — the project's core rule — just saved
real effort. A negative result IS a result: it tells us **cross-adjacent-layer**
redundancy is not the lever.

## What this rules out vs leaves open
- RULED OUT: adjacent-layer delta coding (this probe).
- STILL OPEN (untested, possibly real): cross-layer SHARED codebooks (layers may
  share a *vocabulary of weight-vectors* even while being globally orthogonal —
  different from being similar); low-rank-of-the-stack; weight-symmetry/permutation
  structure. These are different hypotheses; each needs its own fast probe.

## Probe 2 — shared vocabulary across layers: GREEN
Even though layers are orthogonal (delta fails), they may share a *vocabulary* of
weight-vectors. Tested: one codebook for all 12 `mlp.c_fc` layers vs 12 per-layer
codebooks:

| | per-layer (12 codebooks) | ONE shared codebook |
|---|---|---|
| NMSE avg | 0.1039 | 0.1056 |
| quality penalty | — | **1.017× (negligible)** |
| codebook overhead | 12× | **1× (12× less)** |

**One codebook serves all layers at ~no quality cost** → the layers share a weight-
vector vocabulary even while being globally orthogonal. On GPT-2 the bits/weight
saving is tiny (2.014 → 2.001) because there are only 12 layers and the codebook is
small relative to the weights. **But the saving scales with depth**: on an 80-layer
model the per-layer codebook overhead is 80× and sharing it is a real bits/weight win
— and it stacks with healing.

## Honest takeaway for "new tech"
- Cross-adjacent-layer DELTA: **dead** (measured, orthogonal layers).
- Cross-layer shared VOCABULARY (one codebook for the whole stack): **works** (1.017×
  penalty, 12× less codebook), and grows with model depth — a real, if modest,
  global-structure lever that per-matrix methods (AQLM/QuIP#) do not exploit.
- This is the seed of genuinely-new tech: a SINGLE model-wide codebook + per-layer
  indices + healing. Next: measure it whole-model at depth (and on a deeper model
  when available) to size the real win.
